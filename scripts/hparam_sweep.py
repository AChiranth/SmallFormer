#!/usr/bin/env python3
import subprocess
import json
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------
# Fixed training parameters
# -------------------------------------------------------------
COMMON_FLAGS = [
    "--tokens", "texts/tokenized/tokens.npy",
    "--offsets", "texts/tokenized/doc_offsets.npy",
    "--tokenizer", "tokenizer.pt",
    "--block_size", "32",
    "--batch_size", "32",
    "--epochs", "1",
    "--dropout", "0.1",
    "--downsample", "100"
]

sweep = []

fixed_config = {
    "d_model": 384,
    "n_heads": 4,
    "d_ff": 768,
    "lr": 1e-3    # <-- Use your best LR from earlier sweep
}

layers_list = [1, 2, 3, 4]

for n_layers in layers_list:
    cfg = fixed_config.copy()
    cfg["n_layers"] = n_layers
    sweep.append(cfg)

print("Total configs:", len(sweep))
results = []


# -------------------------------------------------------------
# Run one configuration of the LOO pipeline
# -------------------------------------------------------------
def run_loo(config):

    tag = f"d{config['d_model']}_h{config['n_heads']}_lr{config['lr']}"
    json_out = f"loo_results_{tag}.json"

    cmd = ["python", "scripts/loo_train.py"] + COMMON_FLAGS + [
        "--d_model", str(config["d_model"]),
        "--n_heads", str(config["n_heads"]),
        "--lr", str(config["lr"]),
        "--n_layers", "1"
    ]

    print("\n====================================================")
    print("Running configuration:", config)
    print("Saving results to:", json_out)
    print("====================================================\n")

    subprocess.run(cmd, check=True)

    # Rename output file
    Path("loo_results.json").rename(json_out)

    # Load scores
    with open(json_out, "r") as f:
        res = json.load(f)

    # Print detailed fold results
    print("\n--- Fold Scores ---")
    for doc, score in res.items():
        print(f"{doc}: {score:.4f}")

    # Compute overall BERTScore
    scores = list(res.values())
    overall_mean = sum(scores) / len(scores)
    print(f"\n>>> Overall mean BERTScore: {overall_mean:.4f}\n")

    return overall_mean


# -------------------------------------------------------------
# Sweep driver
# -------------------------------------------------------------
def main():
    out_path = Path("full_hparam_results.csv")

    for cfg in sweep:
        avg_score = run_loo(cfg)
        cfg["mean_bertscore"] = avg_score

        results.append(cfg)
        pd.DataFrame(results).to_csv(out_path, index=False)

    df = pd.DataFrame(results).sort_values("mean_bertscore", ascending=False)
    print("\nTop Results:")
    print(df.head())

    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
