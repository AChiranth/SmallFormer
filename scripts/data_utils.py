# scripts/data_utils.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StreamDataset(Dataset):
    """1D autoregressive LM dataset from token stream."""
    def __init__(self, tokens, block_size, downsample=1):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        self.ds = downsample
        self.num_items = max(0, (len(self.tokens) - (block_size + 1)) // downsample)

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        idx *= self.ds
        chunk = self.tokens[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y   # Return CPU tensors


def make_dataloader(tokens, block_size, batch_size, downsample=1):
    ds = StreamDataset(tokens, block_size, downsample)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=2
    )


def load_corpus(tokens_path, offsets_path):
    """← You MUST re-add this function"""
    tokens = np.load(tokens_path)
    offsets = np.load(offsets_path)
    return tokens, offsets
