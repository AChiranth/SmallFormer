import torch
import torch.nn as nn
from ResidualConnection import ResidualConnection
from LayerNormalization import LayerNormalization
from AttentionBlock import AttentionBlock
from FeedForwardNetwork import FeedForwardNetwork

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: AttentionBlock, feed_forward_block: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for i in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in layers:
            x = layer(x, mask)
            
        return self.norm(x)