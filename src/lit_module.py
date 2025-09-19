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

    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input_token_ids = batch["input_token_ids"]
        target_token_ids = batch["target_token_ids"]
        mask = batch["mask"]

        output_token_logits = self.model(input_token_ids)

        # reshape for cross_entropy
        output_token_logits = output_token_logits.reshape(-1, output_token_logits.shape[-1])
        target_token_ids = target_token_ids.reshape(-1)
        mask = mask.reshape(-1)

        loss = F.cross_entropy(output_token_logits, target_token_ids, reduction="none")
        loss = (loss * mask).sum() / mask.sum()
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @override
    def train_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        dataset = NumberDataset(self.config.dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size)

    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
    