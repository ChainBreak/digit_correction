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
        self.model = self.create_model()

    def create_model(self):
        """Create the model"""
        return nn.Linear(self.config.input_dim, self.config.output_dim)
        
    def forward(self, x):
        """Forward pass through the network"""
        pass
    
    def training_step(self, batch, batch_idx):
        print(batch)
        exit()
        loss = 0
        return loss

    @override
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = NumberDataset(self.config.dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size)

    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
    