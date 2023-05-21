from transformers import AutoConfig, EvalPrediction, AutoTokenizer, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments, set_seed
from datasets import load_dataset, Dataset
from evaluate import load
import numpy as np
import wandb
import os
import sys

# consts
MODELS_RESULTS_PATH = 'res.txt'
PREDICT_PATH = 'predictions.txt'
OUTPUT_DIR = 'output'
PROJECT = 'anlp_ex01_results'


def fine_tune_models(models_names: list, dataset_name: str, preprocess_func, compute_metrics, train_samples: int = -1,
                     val_samples: int = -1, test_samples: int = -1, num_seeds: int = 1):
    # set wandb project name
    os.environ['WANDB_PROJECT'] = PROJECT

    total_train_time = 0.0
    models_result = {model_name: None for model_name in models_names}

    # run on each model
    for model_name in models_names:
        model_results = []
        # run on the seeds
        for seed in range(num_seeds):
            set_seed(seed)
            model_name_to_save = f'{model_name}_seed{seed}'

            train_args = TrainingArguments(output_dir=OUTPUT_DIR, report_to='wandb', save_strategy="no",
                                           run_name=model_name_to_save)
            # load datasets
            dataset = load_data(dataset_name=dataset_name, train_samples=train_samples, val_samples=val_samples,
                                test_samples=test_samples)

            # load model and tokenizer
            model, tokenizer = load_model(model_name)

            # preprocess dataset splits with the model tokenizer
            train_dataset = preprocess_dataset(dataset['train'], tokenizer, preprocess_func)
            val_dataset = preprocess_dataset(dataset['val'], tokenizer, preprocess_func)

            # fine tune the model
            train_res, trainer = train(model=model, tokenizer=tokenizer, train_dataset=train_dataset,
                                       val_dataset=val_dataset, train_args=train_args,
                                       compute_metrics_fn=compute_metrics)

            # evaluate and save results
            model_path = OUTPUT_DIR + '/' + model_name_to_save

            eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
            total_train_time += train_res.metrics['train_runtime']
            eval_accuracy = eval_metrics['eval_accuracy']
            model_results.append({
                "model_checkpoint": model_path,
                "trainer": trainer,
                "eval_accuracy": eval_accuracy
            })

            # finish the current run
            wandb.finish()

        # Extract the eval_accuracy values from the model_results list
        eval_accuracies = [result["eval_accuracy"] for result in model_results]

        # Compute the mean and standard deviation
        mean_accuracy = np.mean(eval_accuracies)
        std_accuracy = np.std(eval_accuracies)

        # Find the seed corresponding to the maximum eval_accuracy and save the model
        max_accuracy_seed_index = np.argmax(eval_accuracies)
        max_accuracy_model_path = model_results[max_accuracy_seed_index]["model_checkpoint"]
        max_accuracy_model_trainer = model_results[max_accuracy_seed_index]["trainer"]
        max_accuracy_model_trainer.save_model(max_accuracy_model_path)

        models_result[model_name] = {
            "model_path": max_accuracy_model_path,
            "std": std_accuracy,
            "mean": mean_accuracy
        }

    # pick the best model from all the runs
    best_model_name = max(models_result, key=lambda model_name: models_result[model_name]["mean"])
    best_model_path = models_result[best_model_name]["model_path"]
    print(f'Best model was {best_model_name} model')

    # predict test dataset
    labeled_results, predict_runtime = predict(best_model_name, best_model_path, dataset['test'])

    # log results
    log_results_files(models_result, labeled_results, total_train_time, predict_runtime,
                      MODELS_RESULTS_PATH, PREDICT_PATH)


def train(model, tokenizer, train_dataset, val_dataset, train_args, compute_metrics_fn):
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer
    )

    train_res = trainer.train()

    return train_res, trainer


def load_model(model_name, local_pretrained_path=None):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(local_pretrained_path) if local_pretrained_path \
        else AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

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


def preprocess_dataset(dataset: Dataset, tokenizer, preprocess_fn, for_prediction=False):
    tokenized_datasets = dataset.map(preprocess_fn, fn_kwargs={"tokenizer": tokenizer})
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if for_prediction:
        tokenized_datasets = tokenized_datasets.remove_columns(['labels'])
    return tokenized_datasets


def compute_metrics_func(eval_pred: EvalPrediction):
    metric = load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


def predict(model_name, model_path, test_set):
    # prepare the data and models
    model, tokenizer = load_model(model_name, model_path)
    test_dataset = preprocess_dataset(test_set, tokenizer, preprocess_func, for_prediction=True)

    # create a trainer for prediction
    test_args = TrainingArguments(output_dir=OUTPUT_DIR)
    test_args.set_testing(batch_size=1)

    test_trainer = Trainer(
        model=model,
        args=test_args
    )

    # predict 
    test_result = test_trainer.predict(test_dataset)
    test_runtime = test_result.metrics['test_runtime']
    predictions = np.argmax(test_result.predictions, axis=-1)

    # create the results with their original sentences 
    labeled_results = [{'sentence': str(test_set[i]['sentence']), 'label': predictions[i]}
                       for i in range(len(test_set))]

    return labeled_results, test_runtime


def log_results_files(models_results, labeled_results, train_time, predict_time, statistics_file_path,
                      predictions_file_path):
    # create statistics file
    with open(statistics_file_path, 'w') as statistics_file:
        for model_name, model_stats in models_results.items():
            row = f"{model_name}, {model_stats['mean']}\t+-{model_stats['std']}"
            statistics_file.write(row + '\n')

        statistics_file.write(f'train time,{train_time}\n')
        statistics_file.write(f'predict time,{predict_time}')

    # create prediction file
    with open(predictions_file_path, 'w') as predict_file:
        for result in labeled_results:
            row = f"{result['sentence']}###{result['label']}"
            predict_file.write(row + '\n')


def are_args_validate(args):
    for i in range(len(args)):
        arg_to_test = int(args[i])
        if arg_to_test < -1 or arg_to_test == 0:
            return False

    return True


if __name__ == '__main__':
    models = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
    dataset = 'sst2'

    # args
    if len(sys.argv) != 5:
        print("Missing args for the script")
        sys.exit(1)

    # validate args
    validate = are_args_validate(sys.argv[1:])
    if not validate:
        print("args are not in the legal range")
        sys.exit(1)

    num_seeds_arg = int(sys.argv[1])
    num_train_examples = int(sys.argv[2])
    num_val_examples = int(sys.argv[3])
    num_test_examples = int(sys.argv[4])

    fine_tune_models(models, dataset, train_samples=num_train_examples, val_samples=num_val_examples,
                     test_samples=num_test_examples, preprocess_func=preprocess_func,
                     compute_metrics=compute_metrics_func, num_seeds=num_seeds_arg)
