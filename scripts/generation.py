# scripts/generation.py
import torch
from scripts.training_loop import subsequent_mask

def generate(model, idx, block_size, max_new_tokens, device):
    """Autoregressive sampling."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        mask = subsequent_mask(idx_cond.size(1)).to(device)

        with torch.no_grad():
            logits = model(idx_cond, mask)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_tok], dim=1)
    return idx
