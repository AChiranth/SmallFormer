# scripts/sentence_sampling.py
import re, random

def split_into_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 5]


def split_sentence_in_half(sentence):
    words = sentence.split()
    if len(words) < 6:
        return None
    mid = len(words)//2
    return " ".join(words[:mid]), " ".join(words[mid:])


def sample_partial_pairs(text, k=5):
    sents = split_into_sentences(text)
    random.shuffle(sents)
    pairs = []
    for s in sents:
        halves = split_sentence_in_half(s)
        if halves:
            pairs.append(halves)
        if len(pairs) >= k:
            break
    return pairs
