from tqdm.auto import tqdm
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset  # huggingface datasets

dataset = load_dataset("math_dataset", "arithmetic__mul", num_proc=4)
dataset = dataset.map(
    lambda x: {
        "text": (x["question"][2:-3] + " answer " + x["answer"][2:-3]).replace(
            "*-", "* -"
        )
    },
    num_proc=4,
    remove_columns=["question", "answer"],
)
dataset["val"] = dataset.pop("test")
pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
chars = {}
for a in tqdm(dataset["val"]):
    for r, _ in pre_tokenizer.pre_tokenize_str(a["text"]):
        chars[r] = chars.get(r, 0) + 1


def data_iter():
    for a in dataset["val"]:
        yield a["text"]


tokenizer = Tokenizer(WordLevel())
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.post_processor = TemplateProcessing(
    single="$A [SEP]", pair="$A [SEP] $B:1 [SEP]:1", special_tokens=[("[SEP]", 1)]
)
trainer = WordLevelTrainer(vocab_size=len(chars) + 2, special_tokens=["[PAD]", "[SEP]"])

tokenizer.train_from_iterator(data_iter(), trainer=trainer, length=len(dataset["val"]))
filename = os.path.join(os.path.dirname(__file__), "tokenizer.json")
print("Saving tokenizer to", filename, "...")
tokenizer.save(filename)
