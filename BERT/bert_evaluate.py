import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# -------------------------------
# CONFIGURATION
# -------------------------------
CLEAN_DIR = "../texts/cleaned"      # your cleaned corpus
MODEL_NAME = "gpt2"                 # or bert-base-uncased, distilgpt2, etc.
GEN_TOKENS = 120                    # how many tokens to generate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# LOAD DOCUMENTS
# -------------------------------
def load_docs():
    docs = []
    paths = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.txt")))
    print(f"Found {len(paths)} cleaned documents.")
    for p in paths:
        print(f"Loading {p}")
        with open(p, "r", encoding="utf-8") as f:
            docs.append(f.read().strip())
    return docs


# -------------------------------
# GENERATE USING BERT/GPT MODEL
# -------------------------------
def generate_with_model(prompt, tokenizer, model, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# EVALUATE METRICS
# -------------------------------
def evaluate(reference, generated):
    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(reference, generated)["rougeL"].fmeasure

    # BLEU
    bleu = sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=SmoothingFunction().method1
    )

    # BERTScore
    P, R, F1 = score(
        [generated],
        [reference],
        lang="en",
        model_type="bert-base-uncased"
    )

    return {
        "rouge": rouge,
        "bleu": bleu,
        "bert_P": P.mean().item(),
        "bert_R": R.mean().item(),
        "bert_F1": F1.mean().item(),
    }


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Loading documents…")
    docs = load_docs()

    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    all_scores = []

    for i, doc in enumerate(docs):
        print(f"\n==============================")
        print(f"Evaluating document {i+1}/{len(docs)}")
        print("==============================")

        # Use first 200 chars as a generation prompt
        prompt = doc[:200]

        generated = generate_with_model(prompt, tokenizer, model, max_new_tokens=GEN_TOKENS)

        scores = evaluate(doc, generated)
        all_scores.append(scores)

        print("Generated text sample:\n", generated[:200], "...")
        print("Scores:", scores)

    # Averages
    print("\n============ FINAL AVERAGE SCORES ============")
    avg = {k: sum(x[k] for x in all_scores) / len(all_scores) for k in all_scores[0]}
    for metric, val in avg.items():
        print(f"{metric}: {val:.4f}")


if __name__ == "__main__":
    main()
