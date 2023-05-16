from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments
from datasets import load_dataset, Dataset
from evaluate import load
import torch
import numpy as np


def fine_tune_models(models_names: list, dataset_name: str, preprocess_func, compute_metrics, train_samples: int = -1,
                     val_samples: int = -1, test_samples: int = -1, num_seeds: int = None):
    train_args = TrainingArguments(output_dir='outputs', report_to='wandb', save_strategy="no")
    total_train_time = 0.0
    models_result = {model_name: None for model_name in models_names}

    # run on each model
    for model_name in models_names:
        model_results = []
        # run on the seeds
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            # load datasets
            dataset = load_data(dataset_name=dataset_name, train_samples=train_samples, val_samples=val_samples,
                                test_samples=test_samples)

            # load model and tokenizer
            model, tokenizer = load_model(model_name)

            # preprocess dataset splits with the model tokenizer
            train_dataset = preprocess_dataset(dataset['train'], tokenizer, preprocess_func)
            val_dataset = preprocess_dataset(dataset['val'], tokenizer, preprocess_func)

            # fine tune the model
            train_res, trainer = train(model=model, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset, train_args=train_args, 
                               compute_metrics_fn=compute_metrics)

            # evaluate and save results
            eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
            total_train_time += train_res.metrics['train_runtime']
            eval_accuracy = eval_metrics['eval_accuracy']
            model_results.append({
                "model_checkpoint": model,
                "eval_accuracy": eval_accuracy
            })

        # Extract the eval_accuracy values from the model_results list
        eval_accuracies = [result["eval_accuracy"] for result in model_results]

        # Compute the mean and standard deviation
        mean_accuracy = np.mean(eval_accuracies)
        std_accuracy = np.std(eval_accuracies)

        # Find the seed corresponding to the maximum eval_accuracy, due the order of the list we can know ths seed
        max_accuracy_seed = np.argmax(eval_accuracies)
        max_accuracy_model = model_results[max_accuracy_seed]["model_checkpoint"]

        models_result[model_name] = {
            "model": max_accuracy_model,
            "seed": max_accuracy_seed,
            "std": std_accuracy,
            "mean": mean_accuracy
        }

    # pick and save the best model from all the runs
    best_model_name = max(models_result, key=lambda model_name: models_result[model_name]["mean"])
    best_model = models_result[best_model_name]["model"]


def train(model, tokenizer, train_dataset, val_dataset, train_args, compute_metrics_fn):
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
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
    models = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
    dataset = 'sst2'
    train_samples = 100
    num_seeds = 1

    fine_tune_models(models, dataset, train_samples=train_samples, preprocess_func=preprocess_func,
                      compute_metrics=compute_metrics_func, num_seeds=num_seeds)
    
