from torch.utils.data import Dataset
import numpy as np
import os
import torch


class TextDataset(Dataset):
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


def get_datasets(
    data_dir: str,
    block_size: int,
) -> tuple[TextDataset, TextDataset]:
    train_dataset = TextDataset(data_dir, "train", block_size)
    val_dataset = TextDataset(data_dir, "val", block_size)
    return train_dataset, val_dataset


def make_iter(dset: TextDataset, batch_size: int, device: torch.device):
    while True:
        elems = [dset[torch.randint(len(dset), (1,))] for _ in range(batch_size)]
        xs = [x for x, _ in elems]
        ys = [y for _, y in elems]
        batch = [torch.stack(xs), torch.stack(ys)]
        if device.type == "cuda":
            yield [a.pin_memory().to(device, non_blocking=True) for a in batch]
        else:
            yield [a.to(device, non_blocking=True) for a in batch]
