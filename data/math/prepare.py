# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == "__main__":
    dataset = load_dataset("math_dataset", "arithmetic__mul", num_proc=4)
    dataset = dataset.map(
        lambda x: {
            "text": (x["question"][2:-3] + " answer " + x["answer"][2:-3]).replace(
                "*-", "* -"
            )
        },
        num_proc=4,
        remove_columns=["question", "answer"],
    ).shuffle(seed=42)
    dataset["val"] = dataset.pop("test")

    enc = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(os.path.dirname(__file__), "tokenizer.json")
    )

    def process(example):
        x = enc(example["text"], return_tensors="np")
        out = {"ids": x.input_ids, "len": np.array([len(ids) for ids in x.input_ids])}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        batched=True,
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has 102M tokens
    # val has 1.1M tokens

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
