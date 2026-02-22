import torch
import torch.utils.data
import random
from pydantic import BaseModel

from src.tokenizer import Tokenizer

class NumberDatasetConfig(BaseModel):
    dataset_size: int
    num_digits: int
    conditional: bool
    augmentation_prob: float
    token_length: int
    validation: bool

class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, config: NumberDatasetConfig,  tokenizer: Tokenizer):
        self.config = config
        self.tokenizer = tokenizer


    def __len__(self):
        return self.config.dataset_size

    def __getitem__(self, idx):
        num = self.sample_random_number_from_distribution()
        num_str = f"{num:0{self.config.num_digits}d}"
        augmented_num_str = self.augment_number_string(num_str)

        if self.config.conditional:
            if self.config.validation:
                text = f"{augmented_num_str},"
            else:
                text = f"{augmented_num_str},{num_str}"
        else:
            if self.config.validation:
                text = ""
            else:
                text = f"{num_str}"

        token_ids = self.tokenizer.encode(text=text)
        input_token_ids = token_ids[:-1]
        target_token_ids = token_ids[1:]
        
        return {
            "text": text,
            "number": num,
            "input_token_ids": input_token_ids,
            "target_token_ids": target_token_ids, 
        }

    def sample_random_number_from_distribution(self):
        max_number = 10 ** self.config.num_digits - 1
        base_std = max_number / 10

        # Weighted choices: (mean, std) with sample weights
        choices = [
            (max_number * 1 / 3, max_number / 20),       # peak at 1/3
            (max_number * 2 / 3, max_number / 8),   # peak at 2/3
        ]
        weights = [0.5, 0.5]
        mean, std = random.choices(choices, weights=weights, k=1)[0]
        number = random.normalvariate(mean, std)
        number = min(max(number, 0), max_number)

        return int(number)

    def augment_number_string(self, num_str: str) -> str:
        num_edits = random.randint(1, 3)
        for _ in range(num_edits):
            position = random.randint(0, len(num_str) - 1)
            num_char = random.choice(list("0123456789"))
            num_str = num_str[:position] + num_char + num_str[position + 1:] # replace the digit at the position
        return num_str