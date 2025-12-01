# scripts/evaluation.py
import numpy as np
from bert_score import score as bertscore

def evaluate_bertscore(pred, target):
    _, _, F = bertscore([pred], [target], lang="en", verbose=False)
    return float(F.mean())
