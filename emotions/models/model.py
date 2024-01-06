from transformers import AutoModelForSequenceClassification

id2label = {0: "anger", 1: "fear", 2: "joy", 3: "love", 4: "sadness", 5: "surprise"}
label2id = {v: k for k, v in id2label.items()}

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=6, id2label=id2label, label2id=label2id)

