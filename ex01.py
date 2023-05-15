from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments
from datasets import load_dataset
from evaluate import load
import torch
import wandb


def load_data(dataset_name: str, train_samples: int= -1, val_samples: int = -1, test_samples: int = -1, seed: int = None):
    if seed:
        torch.manual_seed(seed)

    train_dataset = load_dataset(dataset_name, split=f"train[:{train_samples}]")
    val_dataset = load_dataset(dataset_name, split=f"validation[:{val_samples}]")
    test_dataset = load_dataset(dataset_name, split=f"test[:{test_samples}]")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }



dataset = load_data('sst2')
print(dataset)
