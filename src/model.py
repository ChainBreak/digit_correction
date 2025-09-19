import torch.nn as nn
import torch
from typing import final

from pydantic import BaseModel

class TransformerModelConfig(BaseModel):
    vocab_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    hidden_dim: int

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

        # self.pos_encoder = PositionalEncoding(self.config.embed_dim, self.config.max_len)

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
        x = self.transformer(x)
        x = self.classifier(x)
        return x