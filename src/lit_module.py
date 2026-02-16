"""
PyTorch Lightning module for digit correction.
"""

from typing import override
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import NumberDataset
from src.model import TransformerModel
from src.tokenizer import Tokenizer

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
        self.tokenizer = Tokenizer()
        self.model = TransformerModel(self.config.model)
        self._val_numbers: list[int] = []

    def forward(self, x, padding_mask, position_indices):
        """Forward pass through the network"""
        return self.model(x, padding_mask, position_indices)

    
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

    def on_validation_epoch_start(self):
        self._val_numbers: list[int] = []

    def validation_step(self, batch, batch_idx):
        input_token_ids = batch["input_token_ids"]
        position_indices = batch["position_indices"]
        padding_mask = batch["padding_mask"]
        
        input_token_ids = self.auto_regress(input_token_ids, padding_mask, position_indices)

        text_batch = self.decode(input_token_ids)

        numbers = [self.text_to_int(text) for text in text_batch]
        self._val_numbers.extend(numbers)

        return {"numbers": numbers}

    def on_validation_epoch_end(self):
        if not self._val_numbers or self.logger is None:
            return
        fig, ax = plt.subplots()
        ax.hist(self._val_numbers, bins=100, range=(0, 1_000_000), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Number")
        ax.set_ylabel("Count")
        ax.set_title("Validation predicted numbers")
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.add_figure(
                "validation/numbers_histogram", fig, self.current_epoch
            )
        plt.close(fig)


    def auto_regress(self, input_token_ids, padding_mask, position_indices):
        for _ in range(self.config.model.max_token_length):
            input_token_ids, position_indices, padding_mask = self.auto_regress_step(input_token_ids, padding_mask, position_indices)
        return input_token_ids

    def auto_regress_step(self, input_token_ids, padding_mask, position_indices):

        output_token_logits = self.model(input_token_ids, padding_mask, position_indices)
        output_token_probs = F.softmax(output_token_logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(output_token_probs, num_samples=1)


        input_token_ids = torch.cat([input_token_ids, next_token], dim=1)
        position_indices = torch.cat([position_indices, position_indices[:, -1:] + 1], dim=1)
        padding_mask = torch.cat([padding_mask, torch.ones_like(padding_mask[:, -1:])], dim=1)

        return input_token_ids, position_indices, padding_mask

    def decode(self, token_id_batch: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode(tokens_ids.tolist()) for tokens_ids in token_id_batch]

    def text_to_int(self, text: str) -> int:
        try:
            return int(text)
        except ValueError:
            return 0 

    @override
    def train_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        dataset = NumberDataset(self.config.dataset_train, tokenizer=self.tokenizer)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size)

    @override
    def val_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        dataset = NumberDataset(self.config.dataset_validation, tokenizer=self.tokenizer)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
    