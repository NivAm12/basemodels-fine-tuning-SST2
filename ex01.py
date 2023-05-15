from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments
from datasets import load_dataset, Dataset
from evaluate import load
import torch
import numpy as np


def fine_tune_models(models_names: list, dataset_name: str, preprocess_func, compute_metrics, train_samples: int = -1, val_samples: int = -1,
                      test_samples: int = -1, num_seeds: int = None):
    train_args = TrainingArguments(output_dir='outputs', report_to='wandb', use_mps_device=True)

    # run on each seed
    for seed in range(num_seeds):
        torch.manual_seed(seed)

        # load datasets
        dataset = load_data(dataset_name=dataset_name, train_samples=train_samples, val_samples=val_samples, test_samples=test_samples)
        
        # run on the models
        for model_name in models_names:
            # load model and tokernizer
            model, tokenizer = load_model(model_name)

            # preprocess dataset splits with the model tokenizer
            train_dataset = preprocess_dataset(dataset['train'], tokenizer, preprocess_func)
            val_dataset = preprocess_dataset(dataset['val'], tokenizer, preprocess_func)
            train_dataset = preprocess_dataset(dataset['test'], tokenizer, preprocess_func)

            # fine tune the model
            train_res, trainer = train(model=model, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset, train_args=train_args, 
                               compute_metrics_fn=compute_metrics)
            
            print(train_res)
            
            
            


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

    return train_res, trainer


def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    return model, tokenizer


def load_data(dataset_name: str, train_samples: int, val_samples: int, test_samples: int):
    train_dataset = load_dataset(dataset_name, split=f"train[:{train_samples}]")
    val_dataset = load_dataset(dataset_name, split=f"validation[:{val_samples}]")
    test_dataset = load_dataset(dataset_name, split=f"test[:{test_samples}]")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }


def preprocess_func(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True)


def preprocess_dataset(dataset, tokenizer, preprocess_fn):
  tokenized_datasets = dataset.map(preprocess_fn, fn_kwargs={"tokenizer": tokenizer}, batched=True, batch_size=None)
  tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence"])
  tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
  tokenized_datasets = tokenized_datasets.with_format("torch")

  return tokenized_datasets


def compute_metrics_func(eval_pred):
    metric = load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    models = ['bert-base-uncased']
    dataset = 'sst2'
    train_samples = 100
    val_samples = 100
    num_seeds = 1

    fine_tune_models(models, dataset, train_samples=train_samples, val_samples=val_samples, preprocess_func=preprocess_func,
                      compute_metrics=compute_metrics_func, num_seeds=num_seeds)
    
