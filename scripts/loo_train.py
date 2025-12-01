# scripts/loo_train.py

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from torch.serialization import add_safe_globals
from scripts.data_utils import load_corpus, make_dataloader
from scripts.training_loop import train_one_epoch
from scripts.sentence_sampling import sample_partial_pairs
from scripts.generation import generate
from scripts.evaluation import evaluate_bertscore

from src.Transformer import build_transformer
from src.BPETokenizer import BPETokenizer
add_safe_globals([BPETokenizer])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_flush(*args, **kwargs):
    """Convenience wrapper to always flush prints for SLURM."""
    print(*args, **kwargs)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=str, default="texts/tokenized/tokens.npy")
    parser.add_argument("--offsets", type=str, default="texts/tokenized/doc_offsets.npy")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.pt")

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()

    device = get_device()
    print_flush("DEVICE:", device)

    # Load corpus + tokenizer
    tokens, offsets = load_corpus(args.tokens, args.offsets)
    tokenizer: BPETokenizer = torch.load(args.tokenizer)

    vocab_size = tokenizer.vocab_size + 3
    print_flush("Tokenizer vocab size:", tokenizer.vocab_size)

    total_docs = len(offsets) - 1
    results = {}

    for doc_idx in range(total_docs):
        print_flush(f"\n=== LOO Fold {doc_idx+1}/{total_docs} ===")

        s, e = offsets[doc_idx], offsets[doc_idx+1]
        heldout_tokens = tokens[s:e]
        train_tokens = np.concatenate([tokens[:s], tokens[e:]])

        loader = make_dataloader(
            train_tokens,
            args.block_size,
            args.batch_size,
            downsample=args.downsample
        )

        model = build_transformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            N=args.n_layers,
            h=args.n_heads,
            d_ff=args.d_ff,
            seq_len=args.block_size,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(1, args.epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, criterion, device)
            print_flush(f"Epoch {ep}/{args.epochs} loss={loss:.4f}")

        heldout_text = tokenizer.decode(heldout_tokens.tolist())
        pairs = sample_partial_pairs(heldout_text, k=5)

        fold_scores = []
        for first_half, second_half in pairs:
            enc = tokenizer.encode(first_half, return_ids=True)
            idx = torch.tensor(enc).unsqueeze(0).to(device)

            out = generate(model, idx, args.block_size, 80, device)
            pred = tokenizer.decode(out[0].tolist())

            score = evaluate_bertscore(pred, second_half)
            fold_scores.append(score)

        results[f"doc_{doc_idx}"] = float(np.mean(fold_scores))

    with open("loo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print_flush("\nSaved LOO results → loo_results.json")


if __name__ == "__main__":
    main()
