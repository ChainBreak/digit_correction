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

        input_token_ids, target_token_ids, position_indices, padding_mask = self.tokenizer.encode(
            text=text,
            target_length=self.config.token_length,
        )
        
        return {
            "text": text,
            "number": num,
            "input_token_ids": torch.tensor(input_token_ids),
            "target_token_ids": torch.tensor(target_token_ids), 
            "position_indices": torch.tensor(position_indices),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }



    def sample_random_number_from_distribution(self):
        max_number = 10 ** self.config.num_digits - 1
        
        middle_number = max_number // 3

        number = random.normalvariate(middle_number, max_number / 10)
        number = min(max(number, 0), max_number)

        return int(number)

    def augment_number_string(self, num_str: str) -> str:
        for _ in range(1):
            position = random.randint(0, len(num_str) - 1)
            num_char = random.choice(list("0123456789"))
            num_str = num_str[:position] + num_char + num_str[position + 1:] # replace the digit at the position
        return num_str