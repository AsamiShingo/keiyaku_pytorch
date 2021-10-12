import pytest
import os
import tensorflow as tf
from keiyakudata import KeiyakuData
from transformersbase import TransformersBase, TransformersTokenizerBase
import transformers

class TransfoermersEmpty(TransformersBase):
    def __init__(self):
        super().__init__("empty", 20)

    def init_model(self, model_dir_path: str) -> None:
        self.inputs = []
        self.inputs.append(tf.keras.layers.Input((self.seq_len,), dtype=tf.int32))
        self.inputs.append(tf.keras.layers.Input((self.seq_len,), dtype=tf.int32))
        self.inputs.append(tf.keras.layers.Input((self.seq_len,), dtype=tf.int32))

        output = tf.keras.layers.Dense(20)(self.inputs[0])
        self.outputs = { "pooler_output" : output }

        self.transformers_model = None

    def download_save(self, model_full_name: str, model_dir_path: str) -> None:
        pass

class TransfoermersTokenizerEmpty(TransformersTokenizerBase):
    def __init__(self):
        super().__init__("empty")

    def init_tokenizer(self, model_dir_path: str) -> None:
        self.tokenizer = None

    def download_save(self, model_full_name: str, model_dir_path: str) -> None:
        pass

@pytest.fixture(scope="session")
def test_transformers_empty():
    model = TransfoermersEmpty()
    model.init_model("empty")    
    return model

@pytest.fixture(scope="session")
def test_transformers_tokenizer_empty():
    tokenizer = TransfoermersTokenizerEmpty()
    tokenizer.init_tokenizer("empty")
    return tokenizer

@pytest.fixture(scope="session")
def test_keiyakudata():
    test_dir = os.path.join(os.path.dirname(__file__), r"data")    

    keiyakudata = KeiyakuData(os.path.join(test_dir, r"test_keiyaku.csv"))
    return keiyakudata
