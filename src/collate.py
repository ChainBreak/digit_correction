import torch
from typing import Any
from src.tokenizer import Tokenizer
from torch.utils.data.dataloader import default_collate

class Collator:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self,batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_token_ids"]) for item in batch)

        prepared_batch = [self.prepare_sample(sample, max_length) for sample in batch]

        # Use the default PyTorch collate function for final collation
        return default_collate(prepared_batch)

    def prepare_sample(self, sample:dict[str,Any], target_length: int) -> dict[str, torch.Tensor]:
        """Prepeand padding and masks to the front of the sample such that they are all the same length"""

        # Find the current length of the input token ids
        current_length = len(sample["input_token_ids"])

        # Calculate the number of padding tokens needed
        padding_needed = target_length - current_length

        # Create the positive mask and position indices
        positive_mask = [0] * current_length
        position_indices = list(range(current_length))

        # Create the padding and negative mask
        padding = [self.tokenizer.pad_token] * padding_needed
        negative_mask = [1] * padding_needed
        blank_position_indices = [0] * padding_needed

        sample["input_token_ids"] = torch.tensor(padding + sample["input_token_ids"])
        if "target_token_ids" in sample:
            sample["target_token_ids"] = torch.tensor(padding + sample["target_token_ids"])
            
        sample["padding_mask"] = torch.tensor(negative_mask + positive_mask, dtype=torch.bool)
        sample["position_indices"] = torch.tensor(blank_position_indices + position_indices)

        return sample   
