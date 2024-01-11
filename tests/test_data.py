import torch
import os
import emotions.data.make_dataset as make_dataset
from tests import _PATH_DATA
import pytest
from datasets import load_from_disk
from transformers import AutoTokenizer
import shutil

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
class TestMakeDataset:

    @pytest.fixture(autouse=True)
    def test_data_setup(self):
        self.raw_path = os.path.join(_PATH_DATA, "raw/")
        self.processed_path = os.path.join(_PATH_DATA, "processed/")
        make_dataset.main()
        yield
        # teardown code, remove raw and processed folders
        shutil.rmtree(self.raw_path)
        shutil.rmtree(self.processed_path)

    def test_data_exists(self):
        # Check that the raw data exists
        assert os.path.exists(self.raw_path), "raw data not found"
        # Check that the processed data exists
        assert os.path.exists(self.processed_path), "processed data not found"

    def test_data_tokenized(self):
        # self.model_name = "distilbert-base-uncased"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dataset = load_from_disk(os.path.join(_PATH_DATA, "processed"))

        # Check that the training data is tokenized and is stored as a list of tensors
        assert torch.is_tensor(self.dataset["train"]["input_ids"][0]), "input_ids should be a tensor"
        assert torch.is_tensor(self.dataset["train"]["attention_mask"][0]), "attention_mask should be a tensor"
        assert torch.is_tensor(self.dataset["train"]["label"]), "label should be a tensor"
        # Check that the validation data is tokenized
        assert torch.is_tensor(self.dataset["validation"]["input_ids"][0]), "input_ids should be a tensor"
        assert torch.is_tensor(self.dataset["validation"]["attention_mask"][0]), "attention_mask should be a tensor"
        assert torch.is_tensor(self.dataset["validation"]["label"]), "label should be a tensor"
        # Check that the test data is tokenized
        assert torch.is_tensor(self.dataset["test"]["input_ids"][0]), "input_ids should be a tensor"
        assert torch.is_tensor(self.dataset["test"]["attention_mask"][0]), "attention_mask should be a tensor"
        assert torch.is_tensor(self.dataset["test"]["label"]), "label should be a tensor"

    def test_labels(self):
        # test that all labels are in the training dataset
        self.dataset = load_from_disk(os.path.join(_PATH_DATA, "processed"))
        assert set(self.dataset["train"]["label"].numpy()) == set(range(6)), "train labels should contain all numbers from 0 to 5"


