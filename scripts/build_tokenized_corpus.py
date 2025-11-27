from pathlib import Path
import json
import numpy as np

# adjust this import if your class name/module name differ
from src.BPETokenizer import BPETokenizer


CLEANED_DIR = Path("texts/cleaned")
OUT_DIR = Path("texts/tokenized")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_cleaned_texts():
    """Read all .txt files in texts/cleaned as a list of strings."""
    texts = []
    filenames = []
    for path in sorted(CLEANED_DIR.glob("*.txt")):
        with path.open("r", encoding="utf-8") as f:
            txt = f.read().strip()
        texts.append(txt)
        filenames.append(path.name)
    return texts, filenames


def train_or_load_tokenizer(texts):
    """
    Use your BPETokenizer implementation here.
    Adapt the constructor / method names to match BPETokenizer.py.
    """
    # EXAMPLE – change to match your real API
    vocab_size = 8000  # or whatever you’ve been using in TokenizerTest.py

    tokenizer = BPETokenizer(vocab_size=vocab_size, add_bos = True, add_eos = True)

    # If your API is called `train`, `fit`, `build_vocab`, etc.,
    # replace this call accordingly.
    tokenizer.fit(texts)

    # If your tokenizer has a `save` method, use it so you can reuse later.
    # For example:
    # tokenizer.save(OUT_DIR / "bpe_tokenizer.json")

    return tokenizer


def build_flat_token_array(tokenizer, texts):
    """
    Option A (recommended for GPT-style training):
    Concatenate all docs into a single 1D array of token IDs.
    Also return doc_offsets so you know where each original doc starts.
    """
    all_tokens = []
    doc_offsets = [0]  # doc_offsets[i] = start index of doc i in all_tokens

    for text in texts:
        # Use whatever method your BPETokenizer exposes:
        #   ids = tokenizer.encode(text)
        #   ids = tokenizer.tokenize(text)
        #   ids = tokenizer.encode_text(text)
        # Replace this with the correct one.
        ids = tokenizer.encode(text, return_ids=True)

        # Optionally append EOS token if your tokenizer defines one.
        # Example attribute name; change to match your implementation.
        # if hasattr(tokenizer, "eos_id"):
        #     ids = ids + [tokenizer.eos_id]

        all_tokens.extend(ids)
        doc_offsets.append(len(all_tokens))

    tokens = np.array(all_tokens, dtype=np.int32)
    doc_offsets = np.array(doc_offsets, dtype=np.int64)
    return tokens, doc_offsets


def build_per_file_arrays(tokenizer, texts, filenames):
    """
    Option B:
    Keep a separate token array per file.
    Saves an .npz with one array per original text file.
    """
    arrays = {}
    for text, fname in zip(texts, filenames):
        ids = tokenizer.encode(text, return_ids = True)  # adjust method name if needed
        arrays[fname] = np.array(ids, dtype=np.int32)
    return arrays


def main():
    texts, filenames = read_cleaned_texts()
    print(f"Loaded {len(texts)} cleaned files from {CLEANED_DIR}")

    tokenizer = train_or_load_tokenizer(texts)
    print("Trained BPE tokenizer on cleaned corpus.")

    # ---- Option A: single long 1D token array (GPT-style) ----
    tokens, doc_offsets = build_flat_token_array(tokenizer, texts)
    np.save(OUT_DIR / "tokens.npy", tokens)
    np.save(OUT_DIR / "doc_offsets.npy", doc_offsets)
    print(f"Saved flat token array to {OUT_DIR / 'tokens.npy'}")
    print(f"Saved doc offsets to {OUT_DIR / 'doc_offsets.npy'}")

    # ---- Option B: one array per file (optional) ----
    per_file = build_per_file_arrays(tokenizer, texts, filenames)
    np.savez(OUT_DIR / "tokens_per_file.npz", **per_file)
    print(f"Saved per-file token arrays to {OUT_DIR / 'tokens_per_file.npz'}")

    # Optionally, also dump filenames list for convenience
    with (OUT_DIR / "filenames.json").open("w", encoding="utf-8") as f:
        json.dump(filenames, f, indent=2)


if __name__ == "__main__":
    main()
