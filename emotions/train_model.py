from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from models.model import ModelLoader
import evaluate
import numpy as np
import wandb
from datasets import load_from_disk
from omegaconf import OmegaConf # omegaconf is used to load the config file with Hydra

# Load the config file
config = OmegaConf.load("config.yaml")
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
        'accuracy': accuracy.compute(predictions=predictions, references=labels),
        'f1_weighted': f1.compute(predictions= predictions, references=labels, average='weighted'),
        'f1_macro': f1.compute(predictions= predictions, references=labels, average='macro')
    }

# Specify the model name
model_name = "distilbert-base-uncased"

# Load the model
model = ModelLoader(model_name).load_model()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize a data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the dataset from disk
dataset = load_from_disk('./data/processed/')

# Define the training arguments
training_args = TrainingArguments(
    output_dir=f"models/{model_name}",
    learning_rate=config.hyperparameters.learning_rate,
    per_device_train_batch_size=config.hyperparameters.batch_size,
    per_device_eval_batch_size=config.hyperparameters.batch_size,
    num_train_epochs=config.hyperparameters.num_train_epochs,
    weight_decay=config.hyperparameters.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",  # enable reporting to W&B
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the best model and tokenizer
trainer.save_model(f"models/{model_name}/best_model/")
tokenizer.save_pretrained(f"models/{model_name}/best_model/")