import os
import glob
import re
import random
from BPETokenizer import BPETokenizer

CLEAN_DIR = os.path.join("..", "texts", "cleaned")  # adjust if needed

# ---------------------------------------------------------
# FAST & CORRECT CORPUS LOADING (BLOCK CHUNKING)
# ---------------------------------------------------------
def load_cleaned_corpus(block_words=5000, max_blocks=None):
    """
    Load cleaned Dickens text and split into large word blocks.
    This is 10-50x faster and far more memory efficient than line-by-line.

    block_words:     how many words per block
    max_blocks:      optional limit for speed (e.g., 400)
    """
    blocks = []

    for path in glob.glob(os.path.join(CLEAN_DIR, "*.txt")):
        print(f"[+] Loading {path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()

        # Split into uniform blocks
        for i in range(0, len(words), block_words):
            block = " ".join(words[i:i + block_words])
            if block.strip():
                blocks.append(block)

    print(f"[✓] Total blocks before sampling: {len(blocks)}")

    # Optional sampling to speed up BPE
    if max_blocks is not None and len(blocks) > max_blocks:
        random.shuffle(blocks)
        blocks = blocks[:max_blocks]
        print(f"[✓] Using {len(blocks)} sampled blocks for BPE training")

    return blocks

# ---------------------------------------------------------
# PRETTY PRINT HELPERS
# ---------------------------------------------------------
def pretty_print_token_info(tokenizer, text):
    print("\n=== SAMPLE TEXT ===")
    print(text)

    tokens = tokenizer.encode(text)
    print("\n=== TOKENS ===")
    print(tokens)

    token_ids = tokenizer.encode(text, return_ids=True)
    print("\n=== TOKEN IDS ===")
    print(token_ids)

    decoded = tokenizer.decode(token_ids)
    print("\n=== DECODED TEXT ===")
    print(decoded)

# ---------------------------------------------------------
# MAIN TEST SCRIPT
# ---------------------------------------------------------
def main():
    print("Loading cleaned corpus…")
    sentences = load_cleaned_corpus(
        block_words=5000,
        max_blocks=400    # adjust or remove if you want to use all text
    )

    print("\nInitializing BPE Tokenizer…")
    tokenizer = BPETokenizer(
        vocab_size=3000,        # adjust as needed
        add_bos=True,
        add_eos=True,
    )

    print("\nTraining tokenizer… (this should be fast!)")
    tokenizer.fit(sentences)

    print("\n=== RESULTS ===")
    print(f"• Number of merges learned: {len(tokenizer.merges)}")
    print(f"• Final vocab size: {len(tokenizer.token2id)}")

    print("\n=== EXAMPLE MERGES (first 20) ===")
    for m in tokenizer.merges[:20]:
        print("   ", m)

    # Test text
    sample = "It was the best of times, it was the worst of times."
    pretty_print_token_info(tokenizer, sample)

    # Save vocab for debugging
    vocab_path = "bpe_vocab_debug.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok, idx in sorted(tokenizer.token2id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{tok}\n")

    print(f"\nSaved vocab to {vocab_path}")
    print("Tokenizer testing complete.")

# Run
if __name__ == "__main__":
    main()
