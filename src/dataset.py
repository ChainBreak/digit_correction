import torch
import torch.utils.data
import random
from pydantic import BaseModel
from src.tokenizer import Tokenizer
import src.manipulation as manipulation_module
import dataclasses

class NumberDatasetConfig(BaseModel):
    dataset_size: int
    num_digits: int
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
        
        augmented_num_str, undo_command = self.augment_number_string(num_str)

        
        if self.config.validation:
            return NumberTask(
                starting_str=augmented_num_str,
                true_str=f"{num}",
                current_str=augmented_num_str,
            )
        else:
            text = f"{augmented_num_str}|{undo_command}"
       
            token_ids = self.tokenizer.encode(text=text)
            tokens = token_ids[:-1]
            target_tokens = token_ids[1:]
            
            return {
                "text": text,
                "number": num,
                "tokens": tokens,
                "target_tokens": target_tokens, 
            }

    def sample_random_number_from_distribution(self):
        max_number = 10 ** self.config.num_digits - 1
        
        # Weighted choices: (mean, std) with sample weights
        choices = [
            (max_number * 0.25, max_number / 20),
            (max_number * 0.5, max_number / 8),
        ]
        weights = [0.5, 0.5]
        mean, std = random.choices(choices, weights=weights, k=1)[0]
        number = random.normalvariate(mean, std)
        number = min(max(number, 0), max_number)

        return int(number)

    def augment_number_string(self, num_str: str) -> tuple[str, str]:
        
        # F stands for finish
        undo_command = "F"
        number_of_manipulations = random.randint(0, 6 )

        for _ in range(number_of_manipulations):
            num_str, undo_command = self.apply_random_manipulation(num_str)
        
        return num_str, undo_command

    def apply_random_manipulation(self, num_str: str) -> tuple[str, str]:
        command_choices = ["I"]
        if len(num_str) > 0:
            command_choices += ["E", "D"]
            
        match random.choice(command_choices):
            case "E":
                index = random.randint(0, len(num_str) - 1)
                char = random.choice(list("0123456789"))
                command = f"E{index},{char}"
            case "D":
                index = random.randint(0, len(num_str) - 1)
                command = f"D{index}"
            case "I":
                index = random.randint(0, len(num_str))
                char = random.choice(list("0123456789"))
                command = f"I{index},{char}"
            case _:
                command = ""

        manipulated_num_str = manipulation_module.manipulate(num_str, command)
        undo_command = manipulation_module.get_opposite_command(num_str, command)

        return manipulated_num_str, undo_command

@dataclasses.dataclass
class NumberTask:
    starting_str: str
    true_str: str
    current_str: str