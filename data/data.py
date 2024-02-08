from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
import webdataset as wds
from config import Config


class _PreprocessFn:
    def __init__(self, block_size: int):
        self.block_size = block_size

    def __call__(self, gen):
        for sample in gen:
            ids = np.frombuffer(sample["ids"], dtype=np.uint16).astype(int)
            ids = torch.from_numpy(ids)
            # eg len(ids) = 8193, block_size = 1024, len(ids) - block_size = 8193 - 1024 = 7169
            # i will be 0, 1024, 2048, 3072, 4096, 5120, 6144, 7168
            # we'll hit the last index, but not go over
            # eg len(ids) = 8192, block_size = 1024, len(ids) - block_size = 8191 - 1024 = 7168
            # i will be 0, 1024, 2048, 3072, 4096, 5120, 6144
            # we'll skip the last 1023-ish elements
            for i in range(0, len(ids) - self.block_size, self.block_size):
                x = ids[i : i + self.block_size]
                y = ids[i + 1 : i + self.block_size + 1]
                assert len(x) == len(y) == self.block_size, (
                    len(x),
                    len(y),
                    self.block_size,
                )
                yield (x, y)


def _repeatedly(source):
    while True:
        for sample in source:
            yield sample


def _collation_fn(samples):
    res = list(zip(*samples))
    for i in range(len(res)):
        res[i] = torch.stack(res[i])
    return res


def _owt_exception_handler(ext):
    print(f"WDS Exception: {ext}, continuing")


def get_owt_dataset(split: str, batch_size: int, block_size: int, shuffle: bool):
    shards = "{0000..0036}" if split == "train" else "0000"
    base = os.environ.get("OWT_URL", "https://r2.aphoh.us")
    url = f"{base}/openwebtext/{split}/shard-{shards}.tar.gz"

    pipeline = (
        [wds.SimpleShardList(url), _repeatedly]
        + ([wds.split_by_worker, wds.split_by_node] if split == "train" else [])
        + ([wds.shuffle(bufsize=10, initial=4)] if shuffle else [])
        + [
            wds.tarfile_to_samples(handler=_owt_exception_handler),
            _PreprocessFn(block_size),
            _repeatedly,
        ]
        + ([wds.split_by_worker, wds.split_by_node] if split == "val" else [])
        + ([wds.shuffle(bufsize=10000, initial=1000)] if shuffle else [])
        + [wds.batched(batch_size, collation_fn=_collation_fn)]
    )
    n_tokens = 9031397883 if split == "train" else 4432413
    _, world_size, _, num_workers = wds.utils.pytorch_worker_info()
    n_batches = n_tokens // (block_size + 1) // batch_size // world_size // num_workers

    # owt is
    dataset = wds.DataPipeline(*pipeline)
    return dataset


def _get_owt_iter(
    split: str,
    batch_size: int,
    block_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int = 8,
):
    dataset = get_owt_dataset(split, batch_size, block_size, shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    while True:
        for batch in dataloader:
            yield [x.to(device, non_blocking=True) for x in batch]


class _TextDataset(Dataset):
    def __init__(self, data_dir, split="train", block_size=128):
        super().__init__()
        self.data = np.memmap(
            os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r"
        )
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        data = torch.from_numpy(
            (self.data[idx : idx + self.block_size + 1]).astype(np.int64)
        )
        return (
            data[: self.block_size],
            data[1 : 1 + self.block_size],
        )


def _get_binary_datasets(
    data_dir: str,
    block_size: int,
) -> tuple[_TextDataset, _TextDataset]:
    train_dataset = _TextDataset(data_dir, "train", block_size)
    val_dataset = _TextDataset(data_dir, "val", block_size)
    return train_dataset, val_dataset


def _make_iter(dset: _TextDataset, batch_size: int, device: torch.device):
    while True:
        elems = [dset[torch.randint(len(dset), (1,))] for _ in range(batch_size)]
        xs = [x for x, _ in elems]
        ys = [y for _, y in elems]
        batch = [torch.stack(xs), torch.stack(ys)]
        if device.type == "cuda":
            yield [a.pin_memory().to(device, non_blocking=True) for a in batch]
        else:
            yield [a.to(device, non_blocking=True) for a in batch]


def get_dataset_iters(cfg: Config, device: torch.device):
    data_dir = os.path.join(cfg.data_dir, cfg.dataset)

    train_iter, val_iter = None, None
    if cfg.dataset == "openwebtext":
        train_iter = _get_owt_iter(
            "train",
            cfg.micro_batch_size,
            cfg.gpt.block_size,
            True,
            device,
            cfg.num_workers,
        )
        val_iter = _get_owt_iter(
            "val", cfg.micro_batch_size, cfg.gpt.block_size, False, device, 1
        )
        assert (
            cfg.gpt.vocab_size >= 50257
        ), f"vocab size {cfg.gpt.vocab_size} too small for GPT-2"
    else:
        train_dataset, val_dataset = _get_binary_datasets(
            data_dir,
            cfg.gpt.block_size,
        )
        train_iter = _make_iter(train_dataset, cfg.micro_batch_size, device)
        val_iter = _make_iter(val_dataset, cfg.micro_batch_size, device)

    return train_iter, val_iter
