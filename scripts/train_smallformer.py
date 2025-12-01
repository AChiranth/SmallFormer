# scripts/train_smallformer.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataset import make_sequences
from src.Transformer import build_transformer
import time

def train_smallformer(train_tokens, vocab_size, block_size,
                      d_model, n_layers, n_heads, d_ff, dropout,
                      lr, batch_size, epochs, device):

    X, y = make_sequences(train_tokens, block_size)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)

    model = build_transformer(
        vocab_size=vocab_size,
        seq_len=block_size,
        d_model=d_model,
        N=n_layers,
        h=n_heads,
        dropout=dropout,
        d_ff=d_ff
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        t0 = time.time()
        losses = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            mask = None
            out = model.linear(model.decode(xb, mask))
            logits = out.reshape(-1, out.size(-1))
            yb = yb.reshape(-1)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses += loss.item()

        print(f"Epoch {ep+1}/{epochs} Loss={losses/len(loader):.4f} Time={time.time()-t0:.1f}s")

    return model
