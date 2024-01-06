from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from typing import Union, List, Dict


class Predictor:
    """
    A class that loads a pretrained model and tokenizer and uses them to classify the input data.

    Attributes:
        model_name (str): The name of the model to load.
    """

    def __init__(self, model_name: str):
        """
        The constructor for Predictor class.

        Parameters:
            model_name (str): The name of the model to load.
        """
        self.model_name = model_name
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(f"models/{self.model_name}/best_model/")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"models/{self.model_name}/best_model/")
        self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(self, data: Union[str, List[str]]) -> List[Dict[str, float]]:
        """
        Load a pretrained model and tokenizer and use them to classify the input data.

        Parameters:
            data (Union[str, List[str]]): The data to classify. Can be a single string or a list of strings.

        Returns:
            The classification results.
        """

        return self.classifier(data)


if __name__ == '__main__':
    # Load the model
    model_name = "distilbert-base-uncased"
    model = Predictor(model_name)
    print(model.predict("I am happy"))
    print(model.predict("I am sad"))