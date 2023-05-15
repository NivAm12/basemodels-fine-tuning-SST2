from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments
from datasets import load_dataset
from evaluate import load
import torch
import wandb


def train(model, tokenizer, train_dataset, val_dataset, train_args, compute_metrics_fn):
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn
    )

    train_res = trainer.train()

    return {
        "train_result": train_res,
        "trainer": trainer
    }
    

def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    return {
        "model": model,
        "tokenizer": tokenizer
    }


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


def preprocess_func(tokenizer, examples):
    return tokenizer(examples["sentence"], truncation=True)


def preprocess_dataset(dataset, tokenizer, preprocess_fn):
  tokenized_datasets = dataset.map(preprocess_fn, fn_kwargs={"tokenizer": tokenizer})
  tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence"])
  tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
  tokenized_datasets = tokenized_datasets.with_format("torch")

  return tokenized_datasets