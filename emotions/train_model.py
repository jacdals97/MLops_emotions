from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from emotions.models.model import ModelLoader
import evaluate
import numpy as np
from datasets import load_from_disk
import hydra
import wandb
from typing import Dict, Tuple, Type
import os
from google.cloud import secretmanager

# Load the config file
from dotenv import load_dotenv

load_dotenv()


def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """
    Fetches a secret from Google Cloud Secret Manager.

    Parameters:
        project_id (str): Google Cloud project ID.
        secret_id (str): The ID of the secret to fetch.
        version_id (str): The version of the secret to fetch. Defaults to "latest".

    Returns:
        str: The secret value.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")


# Load evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute accuracy and F1 score for the model predictions.

    Parameters:
        eval_pred (Tuple[np.ndarray, np.ndarray]): Contains the model predictions and labels.

    Returns:
        Dict[str, float]: A dictionary with the accuracy and F1 score.
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
    cfg: Dict,
    model_loader: Type[ModelLoader] = ModelLoader,
    tokenizer_class: Type[AutoTokenizer] = AutoTokenizer,
    data_collator_class: Type[DataCollatorWithPadding] = DataCollatorWithPadding,
    trainer_class: Type[Trainer] = Trainer,
) -> None:
    """
    The main function that orchestrates the model training process.

    Parameters:
        cfg (Dict): The configuration object.
        model_loader (Type[ModelLoader], optional): The model loader class. Defaults to ModelLoader.
        tokenizer_class (Type[AutoTokenizer], optional): The tokenizer class. Defaults to AutoTokenizer.
        data_collator_class (Type[DataCollatorWithPadding], optional): The data collator class. Defaults to DataCollatorWithPadding.
        trainer_class (Type[Trainer], optional): The trainer class. Defaults to Trainer.
    """
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


def load_model(model_loader: Type[ModelLoader], model_name: str):
    """
    Loads a model using the provided model loader and model name.

    Parameters:
        model_loader (Type[ModelLoader]): The model loader class.
        model_name (str): The name of the model to load.

    Returns:
        The loaded model.
    """
    return model_loader(model_name).load_model()


def load_tokenizer(tokenizer_class: Type[AutoTokenizer], model_name: str) -> AutoTokenizer:
    """
    Loads a tokenizer using the provided tokenizer class and model name.

    Parameters:
        tokenizer_class (Type[AutoTokenizer]): The tokenizer class.
        model_name (str): The name of the model to load the tokenizer for.

    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    return tokenizer_class.from_pretrained(model_name)


def load_data_collator(data_collator_class: Type[DataCollatorWithPadding], tokenizer: str) -> DataCollatorWithPadding:
    """
    Loads a data collator using the provided data collator class and tokenizer.

    Parameters:
        data_collator_class (Type[DataCollatorWithPadding]): The data collator class.
        tokenizer (str): The tokenizer to use with the data collator.

    Returns:
        DataCollatorWithPadding: The loaded data collator.
    """
    return data_collator_class(tokenizer=tokenizer)


def load_dataset(path: str) -> Dict:
    """
    Loads a dataset from the provided path.

    Parameters:
        path (str): The path to the dataset.

    Returns:
        Dict: The loaded dataset.
    """
    return load_from_disk(path)


def get_training_args(cfg: Dict, save_path: str) -> TrainingArguments:
    """
    Gets the training arguments based on the provided configuration and save path.

    Parameters:
        cfg (Dict): The configuration object.
        save_path (str): The path to save the model.

    Returns:
        TrainingArguments: The training arguments.
    """
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


def initialize_trainer(
    trainer_class: Type[Trainer],
    model: str,
    training_args: TrainingArguments,
    dataset: Dict,
    tokenizer: str,
    data_collator: DataCollatorWithPadding,
) -> Trainer:
    """
    Initializes a trainer with the provided parameters.

    Parameters:
        trainer_class (Type[Trainer]): The trainer class.
        model (str): The model to train.
        training_args (TrainingArguments): The training arguments.
        dataset (Dict): The dataset to train on.
        tokenizer (str): The tokenizer to use.
        data_collator (DataCollatorWithPadding): The data collator to use.

    Returns:
        Trainer: The initialized trainer.
    """
    return trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def train_model(trainer: Trainer) -> None:
    """
    Trains the model using the provided trainer.

    Parameters:
        trainer (Trainer): The trainer to use for training.
    """
    trainer.train()


def save_model_and_tokenizer(trainer: Trainer, tokenizer: str, save_path: str, run: wandb.run) -> None:
    """
    Saves the model and tokenizer using the provided trainer, tokenizer, save path, and run.

    Parameters:
        trainer (Trainer): The trainer to use for saving the model.
        tokenizer (str): The tokenizer to save.
        save_path (str): The path to save the model and tokenizer.
        run (wandb.run): The run to log the artifact.
    """
    trainer.save_model(f"{save_path}/best_model/")
    tokenizer.save_pretrained(f"{save_path}/best_model/")
    artifact = wandb.Artifact(name="best_model", type="model")
    artifact.add_dir(local_path=f"{save_path}/best_model/")  # Add dataset directory to artifact
    run.log_artifact(artifact)


if __name__ == "__main__":
    project_id = "emotions-410912"
    secret_id = "WANDB_API_KEY"
    wandb_api_key = get_secret(project_id, secret_id)
    os.environ["WANDB_API_KEY"] = wandb_api_key
    main()
