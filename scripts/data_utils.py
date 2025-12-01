# scripts/data_utils.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StreamDataset(Dataset):
    """1D autoregressive LM dataset from token stream."""
    def __init__(self, tokens, block_size):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        self.num_items = len(tokens) - (block_size + 1)

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx+self.block_size+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def make_dataloader(tokens, block_size, batch_size, device, downsample=1):
    class StreamDataset(Dataset):
        def __init__(self, tokens):
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            self.block = block_size
            self.ds = downsample

        def __len__(self):
            return max(0, (len(self.tokens) - (self.block + 1)) // self.ds)

        def __getitem__(self, idx):
            idx = idx * self.ds
            chunk = self.tokens[idx : idx + self.block + 1]
            x = chunk[:-1]
            y = chunk[1:]
            return x.to(device), y.to(device)

    ds = StreamDataset(tokens)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def load_corpus(tokens_path, offsets_path):
    """Load token array + doc offsets."""
    tokens = np.load(tokens_path)
    offsets = np.load(offsets_path)
    return tokens, offsets
