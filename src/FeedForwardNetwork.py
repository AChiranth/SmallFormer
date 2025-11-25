import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.ffn(x)