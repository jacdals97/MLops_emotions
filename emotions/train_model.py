from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, WandbCallback
from models.model import ModelLoader
import evaluate
import numpy as np
import wandb
from datasets import load_from_disk

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy(labels, predictions),
        'f1_weighted': f1(labels, predictions, average='weighted'),
        'f1_macro': f1(labels, predictions, average='macro')
    }

model_name = "distilbert-base-uncased"

model = ModelLoader(model_name).load_model()

tokenizer = AutoTokenizer.from_pretrained(model_name)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset = load_from_disk('./data/processed/')

training_args = TrainingArguments(
    output_dir=f"models/{model_name}",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",  # enable reporting to W&B
    run_name=model_name
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[WandbCallback]  # W&B integration
)

trainer.train()

trainer.save_model(f"models/{model_name}/best_model/")
tokenizer.save_pretrained(f"models/{model_name}/best_model/")

