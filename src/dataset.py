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

class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, config: NumberDatasetConfig):
        self.config = config
        self.tokenizer = Tokenizer()


    def __len__(self):
        return self.config.dataset_size

    def __getitem__(self, idx):
        num = self.sample_random_number_from_distribution()
        num_str = f"{num:0{self.config.num_digits}d}"

        if self.config.conditional:
            text = f"[{num_str},{num_str}]"
        else:
            text = f"[{num_str}]"

        token, mask = self.tokenizer.encode(text, target_length=self.config.token_length)
        token = torch.tensor(token)
        mask = torch.tensor(mask)
        
        return {"text": text, "token": token, "mask": mask}



    def sample_random_number_from_distribution(self):
        return random.randint(0, 10 ** self.config.num_digits - 1)