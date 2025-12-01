#!/usr/bin/env python
# coding: utf-8

"""
Training script for SmallFormer decoder-only Transformer.

Assumptions:
- Tokenized data is stored as a NumPy array of token ids at texts/tokenized/tokens.npy
  (1D stream or 2D [num_seqs, seq_len]).
- You have a build_transformer(...) function defined in src/Transformer.py that returns
  a nn.Module with forward(x, mask=None) -> logits of shape (B, T, vocab_size).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.Transformer import Transformer, build_transformer



# ==========================
# Device selection
# ==========================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================
# Dataset
# ==========================

class NumpyLMDataset(Dataset):
    """
    Language-modeling dataset from a NumPy array of token ids.

    Supports:
    - 1D stream of tokens: arr.shape == (N,)
      -> creates many overlapping (input, target) windows of length block_size.
    - 2D array: arr.shape == (num_seqs, seq_len)
      -> one sample per row, taking the first block_size+1 tokens.
    """

    def __init__(self, npy_path: str | Path, block_size: int):
        super().__init__()
        npy_path = Path(npy_path)
        arr = np.load(npy_path)

        if arr.ndim == 1:
            self.mode = "stream"
            self.tokens = torch.from_numpy(arr.astype("int64"))
            self.num_items = self.tokens.size(0) - (block_size + 1)
        elif arr.ndim == 2:
            self.mode = "rows"
            self.tokens = torch.from_numpy(arr.astype("int64"))
            self.num_items = self.tokens.size(0)
        else:
            raise ValueError(f"Unsupported token array shape {arr.shape}")

        if self.num_items <= 0:
            raise ValueError(
                f"Not enough tokens ({self.tokens.numel()}) "
                f"for block_size={block_size}"
            )

        self.block_size = block_size

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int):
        if self.mode == "stream":
            start = idx
            end = start + self.block_size + 1  # +1 for target shift
            chunk = self.tokens[start:end]     # (block_size + 1,)
            x = chunk[:-1]                     # (block_size,)
            y = chunk[1:]                      # (block_size,)
        else:
            row = self.tokens[idx]
            if row.size(0) <= self.block_size:
                raise ValueError(
                    f"Row {idx} length {row.size(0)} <= block_size {self.block_size}"
                )
            chunk = row[: self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]

        return x, y


def load_data_and_create_datasets(
    npy_path: str | Path,
    block_size: int,
    val_fraction: float = 0.1,
    vocab_size_override: int | None = None,
):
    """
    Load the token array, split into train/val, save temporary .npy files,
    and create NumpyLMDataset for each split.

    Returns:
        train_ds, val_ds, vocab_size
    """
    npy_path = Path(npy_path)
    arr = np.load(npy_path)

    # infer vocab size unless explicitly overridden
    inferred_vocab_size = int(arr.max()) + 1
    vocab_size = vocab_size_override or inferred_vocab_size

    if arr.ndim == 1:
        n_tokens = arr.shape[0]
        split_idx = int(n_tokens * (1.0 - val_fraction))
        train_arr = arr[:split_idx]
        val_arr = arr[split_idx:]
    elif arr.ndim == 2:
        n_rows = arr.shape[0]
        split_idx = int(n_rows * (1.0 - val_fraction))
        train_arr = arr[:split_idx]
        val_arr = arr[split_idx:]
    else:
        raise ValueError(f"Unsupported npy shape {arr.shape}")

    out_dir = npy_path.parent
    train_tmp = out_dir / "train_tmp.npy"
    val_tmp = out_dir / "val_tmp.npy"
    np.save(train_tmp, train_arr)
    np.save(val_tmp, val_arr)

    train_ds = NumpyLMDataset(train_tmp, block_size=block_size)
    val_ds = NumpyLMDataset(val_tmp, block_size=block_size)

    return train_ds, val_ds, vocab_size


# ==========================
# Masking
# ==========================

def subsequent_mask(size: int) -> torch.Tensor:
    """
    Create a standard decoder-only casual mask (allow looking at self and previous
    positions, but not future ones).

    Returns a mask of shape (1, 1, size, size) with 1 where allowed, 0 where masked.
    """
    # upper triangular (k=1) are future positions; we set them to 0
    attn_shape = (1, size, size)
    subsequent_mask = torch.tril(torch.ones(attn_shape, dtype=torch.uint8))
    # (1, 1, T, T) for multi-head broadcasting
    return subsequent_mask.unsqueeze(1)  # dtype uint8 / bool


# ==========================
# Train / eval loops
# ==========================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = 1.0,
):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)  # (B, T)
        y = y.to(device)  # (B, T)
        B, T = x.shape

        # build causal mask for this sequence length
        mask = subsequent_mask(T).to(device)  # (1, 1, T, T)

        optimizer.zero_grad(set_to_none=True)

        # forward
        logits = model(x, mask)  # expected (B, T, V)
        B, T, V = logits.shape

        loss = criterion(
            logits.view(B * T, V),
            y.view(B * T),
        )
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * (B * T)
        total_tokens += (B * T)

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        B, T = x.shape

        mask = subsequent_mask(T).to(device)

        logits = model(x, mask)
        B, T, V = logits.shape

        loss = criterion(
            logits.view(B * T, V),
            y.view(B * T),
        )

        total_loss += loss.item() * (B * T)
        total_tokens += (B * T)

    return total_loss / max(total_tokens, 1)


# ==========================
# Main
# ==========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens_path",
        type=str,
        default="texts/tokenized/tokens.npy",
        help="Path to .npy file of token ids.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Sequence length (number of tokens) for each training sample.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (set <=0 to disable).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--pad_idx",
        type=int,
        default=0,
        help="Padding token index (used for ignore_index in loss).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Optional explicit vocab size; if None, inferred from data.",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="checkpoints/smallformer.pt",
        help="Path to save best model checkpoint.",
    )
    # Hyperparameters for your SmallFormer (must match build_transformer signature)
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Transformer hidden size.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of decoder blocks.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=1024,
        help="Feedforward layer dimension.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate.",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # ---- data ----
    train_ds, val_ds, inferred_vocab_size = load_data_and_create_datasets(
        npy_path=args.tokens_path,
        block_size=args.block_size,
        val_fraction=args.val_fraction,
        vocab_size_override=args.vocab_size,
    )

    vocab_size = args.vocab_size or inferred_vocab_size
    print(f"Inferred vocab size: {inferred_vocab_size}, using vocab_size={vocab_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- model ----
    model = build_transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        N=args.n_layers,
        h=args.n_heads,
        d_ff=args.d_ff,
        seq_len=args.block_size,
        dropout=args.dropout,
    )
    model = model.to(device)
    print(model)

    # ---- loss & optimizer ----
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    ckpt_path = Path(args.model_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        )
        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  -> New best model, saving to {ckpt_path}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": vars(args),
                    "vocab_size": vocab_size,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
