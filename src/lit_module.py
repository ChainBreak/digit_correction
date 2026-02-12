"""
PyTorch Lightning module for digit correction.
"""

from typing import override
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import NumberDataset
from src.model import TransformerModel

class DigitCorrectionLitModule(L.LightningModule):
    """
    PyTorch Lightning module for digit correction tasks.
    
    This is a placeholder implementation that can be extended for specific
    digit correction use cases.
    """
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = TransformerModel(self.config.model)

    def forward(self, x, padding_mask, position_indices):
        """Forward pass through the network"""
        return self.model(x, padding_mask, position_indices)

    def auto_regress(self, x):
        """Auto-regressive decoding"""
        for i in range(x.shape[1]):
            output_token_logits = self.model(x)
            next_token = torch.argmax(output_token_logits[:, i, :], dim=-1)
            x = torch.cat([x, next_token.unsqueeze(1)], dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        input_token_ids = batch["input_token_ids"]
        target_token_ids = batch["target_token_ids"]
        position_indices = batch["position_indices"]
        padding_mask = batch["padding_mask"]
        output_token_logits = self.model(input_token_ids, padding_mask, position_indices)

        # reshape for cross_entropy
        output_token_logits = output_token_logits.reshape(-1, output_token_logits.shape[-1])
        target_token_ids = target_token_ids.reshape(-1)
        padding_mask = padding_mask.reshape(-1)

        loss = F.cross_entropy(output_token_logits, target_token_ids, reduction="none")
        loss = (loss * padding_mask).sum() / padding_mask.sum()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    @override
    def train_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        dataset = NumberDataset(self.config.dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size)

    @override
    def val_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        dataset = NumberDataset(self.config.dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
    