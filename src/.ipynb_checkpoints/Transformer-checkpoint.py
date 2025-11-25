import torch
import torch.nn as nn
from InputEmbedding import InputEmbedding
from PositionalEncoding import PositionalEncoding
from AttentionBlock import AttentionBlock
from FeedForwardNetwork import FeedForwardNetwork
from Decoder import DecoderBlock, Decoder
from LanguageModelHead import LanguageModelHead


class Transformer(nn.Module):

    def __init__(self, decoder: Decoder, embedding: InputEmbedding, positional: PositionalEncoding, lmhead: LanguageModelHead):
        self.decoder = decoder
        self.embedding = embedding
        self.positional = positional
        self.lmhead = lmhead

    def decode(self, x, mask):
        x = self.embedding(x)
        x = self.positional(x)
        return self.decoder(x, mask)

    def linear(self, x):
        return self.lmhead(x)



def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 1, h: int = 1, dropout: float = 0.1, d_ff: int = 2048):
    #Create embedding layer
    embedding = InputEmbedding(d_model, vocab_size)

    #Create positional encoding
    positional = PositionalEncoding(d_model, seq_len, dropout)

    #Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = AttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_block = Decoder(decoder_self_attention, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #Create Decoder
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #Create LM Head
    lm_head = LanguageModelHead(d_model, vocab_size)

    #Create Transformer
    transformer = Transformer(decoder, embedding, positional, lm_head)

    #Initialize Transformer parameters to avoid random start and long training
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer