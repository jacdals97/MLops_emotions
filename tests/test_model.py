import torch
from emotions.models.model import ModelLoader
import pytest

class TestModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_name = "distilbert-base-uncased"
        self.model = ModelLoader(self.model_name).load_model()

    def test_model_exists(self):
        assert self.model is not None, "model not found"

    def test_model_output(self):
        # Check that the model outputs a tensor
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        output = self.model(input_ids, attention_mask)
        assert torch.is_tensor(output.logits), "model output should be a tensor"
        assert output.logits.shape == torch.Size([1, 6]), "model output should be a tensor of shape (1, 6)"
        assert output.logits.sum().item() != 0, "model output should not be all zeros"