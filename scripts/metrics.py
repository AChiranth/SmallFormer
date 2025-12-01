# scripts/metrics.py
from rouge_score import rouge_scorer
import nltk
from bert_score import score as bertscore

nltk.download("punkt", quiet=True)

def compute_rouge_l(pred, ref):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure

def compute_bleu(pred, ref):
    pred_tokens = nltk.word_tokenize(pred)
    ref_tokens = [nltk.word_tokenize(ref)]
    return nltk.translate.bleu_score.sentence_bleu(
        ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25)
    )

def compute_bertscore(preds, refs):
    P, R, F = bertscore(preds, refs, lang="en", verbose=False)
    return float(F.mean())
