from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def predict(
    model_name: str,
    data: str | list[str]
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(f"models/{model_name}/best_model/")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}/best_model/")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier(data)
