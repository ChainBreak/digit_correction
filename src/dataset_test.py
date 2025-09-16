from src.dataset import NumberDataset, NumberDatasetConfig


def test_dataset():
    config = NumberDatasetConfig(
        dataset_size=100,
        num_digits=3,
        conditional=True,
        augmentation_prob=0.5,
        token_length=10
    )
    dataset = NumberDataset(config)
    assert len(dataset) == 100
    for i in range(10):
        sample = dataset[i]
        print(sample["text"])
        print(sample["input_token_ids"])
        print(sample["target_token_ids"])
        print(sample["mask"])