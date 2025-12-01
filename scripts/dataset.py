# scripts/dataset.py
import numpy as np
import torch

class TokenizedCorpus:
    def __init__(self, token_path="texts/tokenized/tokens.npy",
                 offset_path="texts/tokenized/doc_offsets.npy"):
        self.tokens = np.load(token_path)
        self.doc_offsets = np.load(offset_path)
        self.num_docs = len(self.doc_offsets) - 1

    def get_document(self, i):
        s = self.doc_offsets[i]
        e = self.doc_offsets[i + 1]
        return self.tokens[s:e]

    def get_train_tokens_excluding(self, i):
        s = self.doc_offsets[i]
        e = self.doc_offsets[i + 1]
        return np.concatenate([self.tokens[:s], self.tokens[e:]])

def make_sequences(tokens, block_size):
    """
    Produces autoregressive training sequences:
    X[t] = tokens[t : t+block_size]
    y[t] = tokens[t+1 : t+block_size+1]
    """
    X, y = [], []
    for i in range(len(tokens) - block_size - 1):
        chunk = tokens[i : i + block_size + 1]
        X.append(chunk[:-1])
        y.append(chunk[1:])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
