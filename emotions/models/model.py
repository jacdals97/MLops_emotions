from transformers import AutoModelForSequenceClassification
from typing import Dict

# Define mappings between labels and ids. These are needed for prediction.
id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label2id = {v: k for k, v in id2label.items()}

# Define a class to load a user-specified model from HuggingFace.
class ModelLoader:
    """
    A class that loads a model for sequence classification from HuggingFace's model hub.

    Attributes:
        model_name (str): The name of the model to load.
    """

    def __init__(self, model_name: str):
        """
        The constructor for ModelLoader class.

        Parameters:
            model_name (str): The name of the model to load.
        """
        self.model_name = model_name

    def load_model(self) -> AutoModelForSequenceClassification:
        """
        The function to load the model from HuggingFace's model hub.

        Returns:
            model (AutoModelForSequenceClassification): A model for sequence classification.
        """
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=6, id2label=id2label, label2id=label2id)