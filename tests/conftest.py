import pytest
import os
from keiyakudata import KeiyakuData
from transfomersbert import TransfomersBert, TransfomersTokenizer

@pytest.fixture(scope="session")
def test_transfomers_bert():
    bert = TransfomersBert()
    bert.init_bert_model(20)
    
    return bert

@pytest.fixture(scope="session")
def test_transfomers_tokenizer():
    tokenizer = TransfomersTokenizer()
    
    return tokenizer

@pytest.fixture(scope="session")
def test_keiyakudata():
    test_dir = os.path.join(os.path.dirname(__file__), r"data")    

    keiyakudata = KeiyakuData(os.path.join(test_dir, r"test_keiyaku.csv"))
    return keiyakudata
