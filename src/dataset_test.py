import pytest

from src.dataset import NumberDataset, NumberDatasetConfig
from src.tokenizer import Tokenizer


@pytest.fixture
def dataset():
    config = NumberDatasetConfig(
        dataset_size=100,
        num_digits=6,
        validation=False,
        token_length=10,
    )
    return NumberDataset(config, tokenizer=Tokenizer())


def test_dataset(dataset: NumberDataset):
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample["text"])


def test_apply_random_manipulation(dataset: NumberDataset):
    num_str = "123456"
    undo_command = "F"
    print(num_str)
    for i in range(12):
        num_str, undo_command = dataset.apply_random_manipulation(num_str)
        print(num_str,undo_command)
  