# scripts/build_tokenizer.py

import json
import torch
from pathlib import Path
from src.BPETokenizer import BPETokenizer

CLEANED_DIR = Path("texts/cleaned")

def load_texts():
    texts = []
    for path in sorted(CLEANED_DIR.glob("*.txt")):
        with path.open("r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    return texts

def main():
    texts = load_texts()

    tokenizer = BPETokenizer(
        vocab_size=8000,
        add_bos=False,
        add_eos=False
    )

    print("Training tokenizer on cleaned texts...")
    tokenizer.fit(texts)

    torch.save(tokenizer, "tokenizer.pt")
    print("Saved tokenizer → tokenizer.pt")

if __name__ == "__main__":
    main()
