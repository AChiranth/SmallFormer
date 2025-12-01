#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import numpy as np

# Ensure we can import src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.BPETokenizer import BPETokenizer


CLEANED_DIR = Path("texts/cleaned")
OUT_DIR = Path("texts/tokenized")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_cleaned_texts():
    texts = []
    filenames = []
    for path in sorted(CLEANED_DIR.glob("*.txt")):
        with path.open("r", encoding="utf-8") as f:
            txt = f.read().strip()
        texts.append(txt)
        filenames.append(path.name)
    return texts, filenames


def build_tokenizer(texts, vocab_size=8000):
    print(f"Training BPETokenizer with vocab_size={vocab_size}")
    tokenizer = BPETokenizer(vocab_size=vocab_size, add_bos=True, add_eos=True)
    tokenizer.fit(texts)

    torch_path = OUT_DIR / "tokenizer.pt"
    import torch
    torch.save(tokenizer, torch_path)
    print(f"Saved tokenizer → {torch_path}")

    return tokenizer


def build_flat_token_array(tokenizer, texts):
    all_tokens = []
    doc_offsets = [0]

    for txt in texts:
        ids = tokenizer.encode(txt, return_ids=True)
        all_tokens.extend(ids)
        doc_offsets.append(len(all_tokens))

    tokens = np.array(all_tokens, dtype=np.int32)
    doc_offsets = np.array(doc_offsets, dtype=np.int64)

    np.save(OUT_DIR / "tokens.npy", tokens)
    np.save(OUT_DIR / "doc_offsets.npy", doc_offsets)

    print(f"Saved tokens.npy ({tokens.shape})")
    print(f"Saved doc_offsets.npy ({doc_offsets.shape})")

    return tokens, doc_offsets


def save_per_file_arrays(tokenizer, texts, filenames):
    per_file = {}
    for txt, name in zip(texts, filenames):
        ids = tokenizer.encode(txt, return_ids=True)
        per_file[name] = np.array(ids, dtype=np.int32)

    np.savez(OUT_DIR / "tokens_per_file.npz", **per_file)
    print("Saved tokens_per_file.npz")


def main():
    texts, filenames = read_cleaned_texts()
    print(f"Loaded {len(texts)} documents from {CLEANED_DIR}")

    tokenizer = build_tokenizer(texts, vocab_size=8000)

    build_flat_token_array(tokenizer, texts)
    save_per_file_arrays(tokenizer, texts, filenames)

    with open(OUT_DIR / "filenames.json", "w") as f:
        json.dump(filenames, f, indent=2)

    print("\nAll preprocessing complete.")


if __name__ == "__main__":
    main()
