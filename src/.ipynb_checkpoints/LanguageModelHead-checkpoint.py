import torch
import torch.nn as nn

class LanguageModelHead(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lmhead = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.lmhead(x), dim = -1)