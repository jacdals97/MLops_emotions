from unittest.mock import Mock, patch
from emotions.train_model import (
    load_model,
    load_tokenizer,
    load_data_collator,
    load_dataset,
    get_training_args,
    initialize_trainer,
    train_model,
    save_model_and_tokenizer,
    compute_metrics
)
from omegaconf import DictConfig
import numpy as np

def test_load_model():
    mock_model_loader = Mock()
    mock_model = Mock()
    mock_model_loader.return_value.load_model.return_value = mock_model

    result = load_model(mock_model_loader, "model_name")
    mock_model_loader.assert_called_once_with("model_name")
    assert result == mock_model


def test_load_tokenizer():
    mock_tokenizer_class = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    result = load_tokenizer(mock_tokenizer_class, "model_name")
    mock_tokenizer_class.from_pretrained.assert_called_once_with("model_name")
    assert result == mock_tokenizer


def test_load_data_collator():
    mock_data_collator_class = Mock()
    mock_data_collator = Mock()
    mock_data_collator_class.return_value = mock_data_collator
    mock_tokenizer = Mock()

    result = load_data_collator(mock_data_collator_class, mock_tokenizer)
    mock_data_collator_class.assert_called_once_with(tokenizer=mock_tokenizer)
    assert result == mock_data_collator


def test_load_dataset():
    with patch("emotions.train_model.load_from_disk") as mock_load_from_disk:
        mock_dataset = Mock()
        mock_load_from_disk.return_value = mock_dataset

        result = load_dataset("path")
        mock_load_from_disk.assert_called_once_with("path")
        assert result == mock_dataset


def test_get_training_args():
    with patch("emotions.train_model.TrainingArguments") as mock_training_arguments:
        mock_args = Mock()
        mock_training_arguments.return_value = mock_args
        cfg = DictConfig(
            {
                "experiment": {
                    "hyperparameters": {
                        "learning_rate": 0.01,
                        "batch_size": 32,
                        "num_train_epochs": 3,
                        "weight_decay": 0.01,
                    },
                    "model_name": "distilbert-base-uncased",
                }
            }
        )

        result = get_training_args(cfg, "save_path")
        mock_training_arguments.assert_called_once()
        assert result == mock_args


def test_initialize_trainer():
    with patch("emotions.train_model.Trainer") as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_model = Mock()
        mock_args = Mock()
        mock_dataset = {"train": Mock(), "validation": Mock()}
        mock_tokenizer = Mock()
        mock_data_collator = Mock()

        result = initialize_trainer(
            mock_trainer_class, mock_model, mock_args, mock_dataset, mock_tokenizer, mock_data_collator
        )
        mock_trainer_class.assert_called_once()
        assert result == mock_trainer

def test_compute_metrics():
    eval_pred = (np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([0, 1]))
    metrics = compute_metrics(eval_pred)
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "f1_macro" in metrics

def test_train_model():
    mock_trainer = Mock()
    train_model(mock_trainer)
    mock_trainer.train.assert_called_once()


def test_save_model_and_tokenizer():
    with patch("emotions.train_model.wandb") as mock_wandb:
        mock_trainer = Mock()
        mock_tokenizer = Mock()
        mock_artifact = Mock()
        mock_wandb.Artifact.return_value = mock_artifact
        mock_run = Mock()
        mock_wandb.run = mock_run

        save_model_and_tokenizer(mock_trainer, mock_tokenizer, "save_path", mock_run)
        mock_trainer.save_model.assert_called_once_with("save_path/best_model/")
        mock_tokenizer.save_pretrained.assert_called_once_with("save_path/best_model/")
        mock_wandb.Artifact.assert_called_once_with(name="best_model", type="model")
        mock_artifact.add_dir.assert_called_once_with(local_path="save_path/best_model/")
        mock_run.log_artifact.assert_called_once_with(mock_artifact)
