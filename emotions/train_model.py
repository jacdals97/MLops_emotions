from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from emotions.models.model import ModelLoader
import evaluate
import numpy as np
from datasets import load_from_disk
import hydra
import wandb

# Load the config file
from dotenv import load_dotenv
from typing import Dict, Tuple


load_dotenv()

# Load evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute accuracy and F1 score for the model predictions.

    Parameters:
        eval_pred transformers.trainer_utils.EvalPrediction: Contains the model predictions and labels.

    Returns:
        A dictionary with the accuracy and F1 score.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "f1_weighted": f1.compute(predictions=predictions, references=labels, average="weighted"),
        "f1_macro": f1.compute(predictions=predictions, references=labels, average="macro"),
    }


@hydra.main(config_path="../config", config_name="default_config.yaml", version_base="1.3")
def main(
    cfg,
    model_loader=ModelLoader,
    tokenizer_class=AutoTokenizer,
    data_collator_class=DataCollatorWithPadding,
    trainer_class=Trainer,
):
    run = wandb.init(reinit=True)
    model_name = cfg.experiment.model_name
    save_path = f"models/{model_name}"
    model = load_model(model_loader, model_name)
    tokenizer = load_tokenizer(tokenizer_class, model_name)
    data_collator = load_data_collator(data_collator_class, tokenizer)
    dataset = load_dataset("./data/processed/")
    training_args = get_training_args(cfg, save_path)
    trainer = initialize_trainer(trainer_class, model, training_args, dataset, tokenizer, data_collator)
    train_model(trainer)
    save_model_and_tokenizer(trainer, tokenizer, save_path, run)


def load_model(model_loader, model_name):
    return model_loader(model_name).load_model()


def load_tokenizer(tokenizer_class, model_name):
    return tokenizer_class.from_pretrained(model_name)


def load_data_collator(data_collator_class, tokenizer):
    return data_collator_class(tokenizer=tokenizer)


def load_dataset(path):
    return load_from_disk(path)


def get_training_args(cfg, save_path):
    return TrainingArguments(
        output_dir=save_path,
        learning_rate=cfg.experiment.hyperparameters.learning_rate,
        per_device_train_batch_size=cfg.experiment.hyperparameters.batch_size,
        per_device_eval_batch_size=cfg.experiment.hyperparameters.batch_size,
        num_train_epochs=cfg.experiment.hyperparameters.num_train_epochs,
        weight_decay=cfg.experiment.hyperparameters.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
    )


def initialize_trainer(trainer_class, model, training_args, dataset, tokenizer, data_collator):
    return trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def train_model(trainer):
    trainer.train()


def save_model_and_tokenizer(trainer, tokenizer, save_path, run):
    trainer.save_model(f"{save_path}/best_model/")
    tokenizer.save_pretrained(f"{save_path}/best_model/")
    artifact = wandb.Artifact(name="best_model", type="model")
    artifact.add_dir(local_path=f"{save_path}/best_model/")  # Add dataset directory to artifact
    run.log_artifact(artifact)


if __name__ == "__main__":
    main()
