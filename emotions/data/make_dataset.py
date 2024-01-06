from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import transformers

if __name__ == '__main__':
    # Load dataset from huggingface
    dataset = load_dataset("dair-ai/emotion", cache_dir="./data/raw")  # use cache_dir to store raw data

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples: datasets.formatting.formatting.LazyBatch) -> transformers.tokenization_utils_base.BatchEncoding:
        """
        Tokenize the text data in the examples.

        Parameters:
            examples datasets.formatting.formatting.LazyBatch: A dictionary containing the text data.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding: A dictionary with the tokenized text data.
        """
        return tokenizer(examples["text"])

    # Tokenize the datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Format the datasets as torch tensors
    tokenized_datasets["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_datasets["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_datasets["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Save the tokenized datasets to disk
    tokenized_datasets.save_to_disk("./data/processed")
