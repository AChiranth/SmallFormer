# scripts/generation.py
import torch

def generate(model, idx, max_new_tokens, block_size):
    """
    idx: (1, T) token tensor
    Returns extended sequence with T + max_new_tokens
    """
    model.eval()
    for _ in range(max_new_tokens):

        # Always feed only last block_size tokens
        if idx.size(1) > block_size:
            idx_cond = idx[:, -block_size:]
        else:
            idx_cond = idx

        mask = None
        with torch.no_grad():
            out = model.linear(model.decode(idx_cond, mask))
            logits = out[:, -1, :]  # last timestep
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

        idx = torch.cat([idx, next_id], dim=1)

    return idx
