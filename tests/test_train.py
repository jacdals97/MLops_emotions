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
    compute_metrics,
)
import numpy as np


def test_load_model():
    model_loader = Mock()
    model_name = "distilbert-base-uncased"
    load_model(model_loader, model_name)
    model_loader.assert_called_once_with(model_name)


def test_load_tokenizer():
    tokenizer_class = Mock()
    model_name = "distilbert-base-uncased"
    load_tokenizer(tokenizer_class, model_name)
    tokenizer_class.from_pretrained.assert_called_once_with(model_name)


def test_load_data_collator():
    data_collator_class = Mock()
    tokenizer = Mock()
    load_data_collator(data_collator_class, tokenizer)
    data_collator_class.assert_called_once_with(tokenizer=tokenizer)


@patch("emotions.train_model.load_from_disk")
def test_load_dataset(mock_load_from_disk):
    path = "./data/processed/"
    load_dataset(path)
    mock_load_from_disk.assert_called_once_with(path)


def test_get_training_args():
    cfg = Mock()
    cfg.hyperparameters.learning_rate = 0.01
    cfg.hyperparameters.batch_size = 16
    cfg.hyperparameters.num_train_epochs = 3
    cfg.hyperparameters.weight_decay = 0.01
    cfg.online = False
    save_path = "models/distilbert-base-uncased"
    training_args = get_training_args(cfg, save_path)
    assert training_args.learning_rate == 0.01
    assert training_args.per_device_train_batch_size == 16
    assert training_args.per_device_eval_batch_size == 16
    assert training_args.num_train_epochs == 3
    assert training_args.weight_decay == 0.01
    assert training_args.output_dir == save_path


@patch("emotions.train_model.Trainer")
def test_initialize_trainer(mock_trainer_class):
    model = Mock()
    training_args = Mock()
    dataset = {"train": Mock(), "validation": Mock()}
    tokenizer = Mock()
    data_collator = Mock()
    initialize_trainer(mock_trainer_class, model, training_args, dataset, tokenizer, data_collator)
    mock_trainer_class.assert_called_once()


def test_compute_metrics():
    eval_pred = (np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([0, 1]))
    metrics = compute_metrics(eval_pred)
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "f1_macro" in metrics


@patch("emotions.train_model.Trainer")
def test_train_model(mock_trainer):
    train_model(mock_trainer)
    mock_trainer.train.assert_called_once()


@patch("emotions.train_model.Trainer")
def test_save_model_and_tokenizer(mock_trainer):
    tokenizer = Mock()
    save_path = "models/distilbert-base-uncased"
    save_model_and_tokenizer(mock_trainer, tokenizer, save_path)
    mock_trainer.save_model.assert_called_once_with(f"{save_path}/best_model/")
    tokenizer.save_pretrained.assert_called_once_with(f"{save_path}/best_model/")
