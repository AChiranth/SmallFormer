from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyLMDataset(Dataset):
    """
    Language-modeling dataset from a 1D NumPy array of token ids.
    Each item is (input_ids, target_ids) where target_ids is shifted by 1.
    """
    def __init__(self, npy_path: str | Path, block_size: int):
        npy_path = Path(npy_path)
        arr = np.load(npy_path)  # expect int token ids

        if arr.ndim == 1:
            self.tokens = torch.from_numpy(arr.astype("int64"))
            self.mode = "stream"
        elif arr.ndim == 2:
            # treat each row as an independent sequence
            self.tokens = torch.from_numpy(arr.astype("int64"))
            self.mode = "rows"
        else:
            raise ValueError(f"Unsupported token array shape {arr.shape}")

        self.block_size = block_size

        if self.mode == "stream":
            # number of possible starting positions for a length block_size + 1 slice
            self.num_items = self.tokens.size(0) - (block_size + 1)
        else:
            # one sample per row (you can make this fancier later)
            self.num_items = self.tokens.size(0)

    def __len__(self) -> int:
        return max(self.num_items, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "stream":
            # take a contiguous window from the stream
            start = idx
            end = start + self.block_size + 1  # +1 for target shift
            chunk = self.tokens[start:end]     # (block_size + 1,)
            x = chunk[:-1]                     # (block_size,)
            y = chunk[1:]                      # (block_size,)
        else:
            # single row: use first (block_size + 1) tokens
            row = self.tokens[idx]
            if row.size(0) <= self.block_size:
                raise ValueError(
                    f"Row {idx} length {row.size(0)} <= block_size {self.block_size}"
                )
            chunk = row[: self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]

        return x, y
