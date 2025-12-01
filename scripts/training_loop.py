# scripts/training_loop.py
import torch
import torch.nn as nn
from tqdm import tqdm

def subsequent_mask(size):
    """Standard causal mask."""
    mask = torch.tril(torch.ones(size, size, dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(0)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip=1.0,
    use_tqdm=True
):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    iterator = tqdm(dataloader, desc="Training", leave=False) if use_tqdm else dataloader

    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)

        B, T = x.shape
        optimizer.zero_grad(set_to_none=True)

        # causal mask
        mask = torch.tril(torch.ones((1, 1, T, T), dtype=torch.uint8, device=device))

        logits = model(x, mask)
        B, T, V = logits.size()

        loss = criterion(
            logits.view(B*T, V),
            y.view(B*T),
        )

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * (B * T)
        total_tokens += (B * T)

        if use_tqdm:
            iterator.set_postfix(loss=loss.item())

    return total_loss / max(total_tokens, 1)
