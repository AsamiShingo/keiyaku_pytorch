import pytest
import os
from keiyakudata import KeiyakuData
from transformersbert import TransformersBert, TransformersTokenizer

@pytest.fixture(scope="session")
def test_transformers_bert():
    bert = TransformersBert()
    bert.init_bert_model(20)
    
    return bert

@pytest.fixture(scope="session")
def test_transformers_tokenizer():
    tokenizer = TransformersTokenizer()
    
    return tokenizer

@pytest.fixture(scope="session")
def test_keiyakudata():
    test_dir = os.path.join(os.path.dirname(__file__), r"data")    

    keiyakudata = KeiyakuData(os.path.join(test_dir, r"test_keiyaku.csv"))
    return keiyakudata
