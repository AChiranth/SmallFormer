# scripts/sentence_sampling.py
import re
import random

def split_into_sentences(text):
    """
    Simple rule-based sentence splitter.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def split_sentence_in_half(sentence):
    """
    Returns (first_half, second_half) based on token count.
    """
    words = sentence.split()
    if len(words) < 6:
        return None  # skip tiny sentences

    mid = len(words) // 2
    first = " ".join(words[:mid])
    second = " ".join(words[mid:])
    return first, second

def sample_partial_completion_pairs(text, k=6):
    """
    Samples k sentences and splits them into halves.
    Returns list[(first_half, second_half)]
    """
    sentences = split_into_sentences(text)
    random.shuffle(sentences)

    pairs = []
    for s in sentences:
        halves = split_sentence_in_half(s)
        if halves:
            pairs.append(halves)
        if len(pairs) >= k:
            break

    return pairs
