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

        self.pos_encoder = PositionEmbedding(
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

    def forward(self, x, padding_mask, position_indices) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x, position_indices)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.classifier(x)
        return x

@final
class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        embedding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        # embedding = embedding.unsqueeze(0)
        self.register_buffer("embedding", embedding)
    
    def forward(self, x: torch.Tensor, position_indices: torch.Tensor) -> torch.Tensor:

    
        # Collect the embeddings at the position indices
        embedding = self.embedding[position_indices]


        x = x + embedding
        return x