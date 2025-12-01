# scripts/loo_evaluate.py
import torch
import json
import numpy as np
import argparse

from dataset import TokenizedCorpus
from train_smallformer import train_smallformer
from generation import generate
from sentence_sampling import sample_partial_completion_pairs
from metrics import compute_rouge_l, compute_bleu, compute_bertscore

from src.BPETokenizer import BPETokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def loo_evaluate(
    vocab_size,
    block_size,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    dropout,
    lr,
    batch_size,
    epochs,
    tokenizer_path="tokenizer.pt"
):

    # Load dataset
    corpus = TokenizedCorpus()
    # Load tokenizer
    tokenizer: BPETokenizer = torch.load(tokenizer_path)

    results = {}

    for doc_idx in range(corpus.num_docs):
        print(f"\n=== LOO Fold {doc_idx+1}/{corpus.num_docs} ===")

        heldout_tokens = corpus.get_document(doc_idx)
        heldout_text = tokenizer.decode(heldout_tokens.tolist())

        # Sample partial completion pairs
        pairs = sample_partial_completion_pairs(heldout_text, k=6)

        train_tokens = corpus.get_train_tokens_excluding(doc_idx)

        # Train model
        model = train_smallformer(
            train_tokens=train_tokens,
            vocab_size=vocab_size,
            block_size=block_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=DEVICE
        )

        fold_scores = {"rougeL": [], "bleu": [], "bertscore": []}

        for first_half, target_half in pairs:
            enc = tokenizer.encode(first_half, return_ids=True)
            idx = torch.tensor(enc, dtype=torch.long).unsqueeze(0).to(DEVICE)

            out = generate(model, idx, max_new_tokens=80, block_size=block_size)
            prediction = tokenizer.decode(out[0].tolist())

            fold_scores["rougeL"].append(compute_rouge_l(prediction, target_half))
            fold_scores["bleu"].append(compute_bleu(prediction, target_half))
            fold_scores["bertscore"].append(compute_bertscore([prediction], [target_half]))

        results[f"doc_{doc_idx}"] = {
            "rougeL": float(np.mean(fold_scores["rougeL"])),
            "bleu": float(np.mean(fold_scores["bleu"])),
            "bertscore": float(np.mean(fold_scores["bertscore"]))
        }

        print("Fold scores:", results[f"doc_{doc_idx}"])

    with open("loo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved results → loo_results.json")


# -------------------- CLI SUPPORT --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.pt")

    args = parser.parse_args()

    loo_evaluate(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        tokenizer_path=args.tokenizer_path
    )
