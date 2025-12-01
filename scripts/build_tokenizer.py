import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.BPETokenizer import BPETokenizer

CLEANED_DIR = ROOT / "texts" / "cleaned"

def load_texts():
    texts = []
    for path in sorted(CLEANED_DIR.glob("*.txt")):
        with path.open("r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    return texts

def main():
    print("Loading cleaned texts…")
    texts = load_texts()
    print(f"Loaded {len(texts)} documents.")

    print("Training BPETokenizer…")
    tokenizer = BPETokenizer(
        vocab_size=8000,
        add_bos=False,
        add_eos=False
    )
    tokenizer.fit(texts)

    save_path = ROOT / "tokenizer.pt"
    torch.save(tokenizer, save_path)

    print(f"Saved tokenizer → {save_path}")

if __name__ == "__main__":
    main()
