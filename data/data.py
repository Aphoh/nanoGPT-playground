from torch.utils.data import Dataset, DataLoader
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


def get_data_loaders(
    data_dir: str, batch_size: int, block_size: int, device: torch.device, num_workers=4
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TextDataset(data_dir, "train", block_size)
    val_dataset = TextDataset(data_dir, "val", block_size)
    attrs = {
        "num_workers": num_workers,
        "shuffle": True,
        "batch_size": batch_size,
    }
    if device.type == "cuda":
        attrs["pin_memory"] = True
        attrs["pin_memory_device"] = device

    train_loader = DataLoader(train_dataset, **attrs)
    val_loader = DataLoader(val_dataset, **attrs)
    return train_loader, val_loader


def make_iter(loader: DataLoader, device: torch.device):
    while True:
        for elem in iter(loader):
            yield [x.to(device, non_blocking=True) for x in elem]
