import torch.nn as nn
import torch
from typing import final
import math

from pydantic import BaseModel

class TransformerModelConfig(BaseModel):
    vocab_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    max_token_length: int

@final
class TransformerModel(nn.Module):
    def __init__(self, config: TransformerModelConfig):
        super().__init__()
        # Validate the config using the BaseModel's validation
        self.config = TransformerModelConfig.model_validate(config)
   
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size, 
            embedding_dim=self.config.embed_dim, 
            )

        self.pos_encoder = PositionalEncoding(
            embed_dim=self.config.embed_dim, 
            max_len=self.config.max_token_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=config.num_layers,
        )

        self.classifier = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
        )

    def forward(self, x) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.classifier(x)
        return x

@final
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.encoding = nn.Parameter(encoding, requires_grad=False)
    
    def forward(self, x):
        input_len = x.shape[1]

        # cut down the encoding to the input length
        encoding = self.encoding[:,:input_len, :]

        x = x + encoding
        return x