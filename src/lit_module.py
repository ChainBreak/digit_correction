"""
PyTorch Lightning module for digit correction.
"""

from collections import defaultdict
from math import e
from typing import Any, override
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import NumberDataset, NumberTask
from src.model import TransformerModel
from src.tokenizer import Tokenizer
from src import collate
from src import manipulation

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
        self.collator = collate.Collator(self.tokenizer)
        self.model = TransformerModel(self.config.model)
        self._validation_results: defaultdict[str, list[int|float]] = defaultdict(list)

    def forward(self, x, padding_mask, position_indices):
        """Forward pass through the network"""
        return self.model(x, padding_mask, position_indices)

    
    def training_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        target_tokens = batch["target_tokens"]
        position_indices = batch["position_indices"]
        padding_mask = batch["padding_mask"]
        loss_mask = 1-padding_mask.float()
        # print("--------------------------------")
        # print(self.tokenizer.decode(tokens[0].tolist()))
        # print(self.tokenizer.decode(target_tokens[0].tolist()))

        output_token_logits = self.model(tokens, padding_mask, position_indices)

        # reshape for cross_entropy
        output_token_logits = output_token_logits.reshape(-1, output_token_logits.shape[-1])
        target_tokens = target_tokens.reshape(-1)
        loss_mask = loss_mask.reshape(-1)

        loss = F.cross_entropy(output_token_logits, target_tokens, reduction="none")
        loss = (loss * loss_mask).sum() / loss_mask.sum()


        self.log("train_loss", loss, prog_bar=True)
        return loss


    def on_validation_epoch_start(self):
        self._validation_results = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        numbers_to_manipulate: list[NumberTask] = batch

        # Pick a number and record the history as it is being manipulated
        history_list = []

        if batch_idx % 10 == 0:
            number_task_to_record = numbers_to_manipulate[0]
        else:
            number_task_to_record = None

        for edit_i in range(self.config.max_edit_steps):

            if len(numbers_to_manipulate) == 0:
                break
            
            prompt_list = [f"{number_task.current_str}|" for number_task in numbers_to_manipulate]
            completed_text_list = self.complete_text(prompt_list)
            next_numbers_to_manipulate = []

            
            # for each completion in the batch
            for text, number_task in zip(completed_text_list, numbers_to_manipulate):
                try:
                    if number_task == number_task_to_record:
                        history_list.append(text)

                    prompt_number_str, response_text = text.split("|")
                    
                    # F is for finished
                    if "F" in response_text:
                        self._validation_results["true_numbers"].append(self.text_to_int(number_task.true_str))
                        self._validation_results["prompt_numbers"].append(self.text_to_int(number_task.starting_str))
                        self._validation_results["output_numbers"].append(self.text_to_int(number_task.current_str))
                        if len(history_list) > 0:
                            history_text = "\n".join(history_list)
                            self.logger.experiment.add_text(
                                "validation/history",
                                history_text,
                                self.current_epoch * 1000 + batch_idx
                            )

                        continue # task is complete, don't add to next_numbers_to_manipulate
                    else:
                        manipulated_number_str = manipulation.manipulate(prompt_number_str, response_text)
                        number_task.current_str = self.sanitize_number_text(manipulated_number_str)
                except Exception as e:
                    pass
                    # print(e,text)
                next_numbers_to_manipulate.append(number_task)

            numbers_to_manipulate = next_numbers_to_manipulate

   
    def on_validation_epoch_end(self):


        fig, axes = plt.subplots(3, 1, figsize=(12, 15))#width, height
        hist_range = (0, 1_000_000)
        bins = 100

        axes[0].hist(self._validation_results["true_numbers"], bins=bins, range=hist_range, alpha=0.5, color="green")
        axes[0].set_xlabel("Number")
        axes[0].set_ylabel("Count")
        axes[0].set_title("True number distribution")

        axes[1].hist(self._validation_results["prompt_numbers"], bins=bins, range=hist_range, alpha=0.5, color="red")
        axes[1].set_xlabel("Number")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Prompt number distribution")


        axes[2].hist(self._validation_results[f"true_numbers"], bins=bins, range=hist_range, alpha=0.5, color="green", label="True numbers")
        axes[2].hist(self._validation_results[f"output_numbers"], bins=bins, range=hist_range, alpha=0.5, color="orange", label="Output numbers")
        axes[2].legend()
        axes[2].set_xlabel("Number")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"Output number distribution")



        if hasattr(self.logger, "experiment"):
            self.logger.experiment.add_figure(
                "validation/numbers_histogram", fig, self.current_epoch
            )
        plt.close(fig)


    def complete_text(self, text_list: list[str]) -> list[str]:

        # convert the text to tokens without the EOS token
        tokens_list = [self.tokenizer.encode(text)[:-1] for text in text_list]

        # convert the text to tokens dict used for the collator
        dataset_items = [{"tokens": tokens} for tokens in tokens_list]

        # collate the tokens into a batch
        batch = self.collator.collate_fn(dataset_items)

        tokens = batch["tokens"].to(self.device)
        padding_mask = batch["padding_mask"].to(self.device)
        position_indices = batch["position_indices"].to(self.device)

        final_tokens = self.auto_regress(tokens, padding_mask, position_indices)

        output_text_list = self.decode_batch(final_tokens)

        return output_text_list

    def auto_regress(self, tokens, padding_mask, position_indices):
        
        for _ in range(self.config.max_auto_regress_steps):
            tokens, position_indices, padding_mask = self.auto_regress_step(tokens, padding_mask, position_indices)
        return tokens

    def auto_regress_step(self, tokens, padding_mask, position_indices):

        # Get the output logits for every every token at every position for the batch
        output_token_logits = self.model(tokens, padding_mask, position_indices)
        
        # Convert the logits to probabilities just for the last tokens of the batch
        output_token_probs = F.softmax(output_token_logits[:, -1, :], dim=-1)
        
        # Sample the next token for the batch
        next_token = torch.multinomial(output_token_probs, num_samples=1)

        # Append the next token to the input token ids and the position indices and the padding mask
        tokens = torch.cat([tokens, next_token], dim=1)
        position_indices = torch.cat([position_indices, position_indices[:, -1:] + 1], dim=1)
        padding_mask = torch.cat([padding_mask, torch.zeros_like(padding_mask[:, -1:])], dim=1)

        return tokens, position_indices, padding_mask

    def decode_batch(self, token_id_batch: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode(tokens_ids.tolist()) for tokens_ids in token_id_batch]

    def text_to_int(self, text: str) -> int:
        try:
            return int(text)
        except ValueError:
            return 0 

    def sanitize_number_text(self, text: str) -> str:
        if text == "":
            return ""
        return f"{int(text)}"

    @override
    def train_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        dataset = NumberDataset(self.config.dataset_train, tokenizer=self.tokenizer)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            collate_fn=self.collator.collate_fn,
            num_workers=self.config.dataset_train.num_workers,
        )

    @override
    def val_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        dataset = NumberDataset(self.config.dataset_validation, tokenizer=self.tokenizer)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=lambda x: x, # no collate needed, just return the batch as a list
            num_workers=self.config.dataset_validation.num_workers,
        )

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.max_epochs, eta_min=0.0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }




