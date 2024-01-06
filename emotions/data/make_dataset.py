from datasets import load_dataset, load_dataset_builder
from datasets import get_dataset_split_names
from transformers import AutoTokenizer

if __name__ == '__main__':
    # Load dataset from huggingface
    dataset = load_dataset("dair-ai/emotion", cache_dir="./data/raw") # use cache_dir to store raw data

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # format dataset as torch
    tokenized_datasets["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_datasets["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_datasets["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # save dataset
    tokenized_datasets.save_to_disk("./data/processed")

    # 
    pass