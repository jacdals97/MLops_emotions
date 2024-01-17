from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from typing import Union, List, Dict
import wandb
import os
from dotenv import load_dotenv

load_dotenv()
entity = os.getenv('WANDB_ENTITY')
project = os.getenv('WANDB_PROJECT')

class Predictor:
    """
    A class that loads a pretrained model and tokenizer and uses them to classify the input data.

    Attributes:
        model_name (str): The name of the model to load.
    """

    def __init__(self, artifact: str):
        """
        The constructor for Predictor class.

        Parameters:
            model_name (str): The name of the model to load.
        """
        self.artifact = artifact

        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{artifact}")
        artifact_dir = artifact.download(f"models/best_model/")

        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
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


if __name__ == "__main__":
    # Load the model
    artifact = "best_model:v1"
    model = Predictor(artifact)
    print(model.predict("I am happy"))
    print(model.predict("I am sad"))
