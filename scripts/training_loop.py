# scripts/training_loop.py
import torch
import torch.nn as nn

def subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size, dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(0)

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip=1.0
):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)

        B, T = x.shape
        optimizer.zero_grad(set_to_none=True)

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

    # return mean loss for this epoch
    return total_loss / max(total_tokens, 1)
