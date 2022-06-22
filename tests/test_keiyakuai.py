import pytest
from keiyakuai import KeiyakuAI
from transformersbert import TransformersBert, TransformersTokenizerBert

class TestKeiyakuAI:

    class TestKeiyakuAIBert:
        @pytest.fixture(scope="class")
        def test_model(self, tmpdir_factory):
            tmpdir = tmpdir_factory.mktemp("test_model_bert")

            model = TransformersBert(seq_len = 20, output_dim=64)
            model.download_save(tmpdir)
            model.init_model(tmpdir)    
            yield model
            model = None

        @pytest.fixture(scope="class")
        def test_tokenizer(self, tmpdir_factory):
            tmpdir = tmpdir_factory.mktemp("test_tokenizer_bert")

            tokenizer = TransformersTokenizerBert()
            tokenizer.download_save(tmpdir)
            tokenizer.init_tokenizer(tmpdir)    
            yield tokenizer
            tokenizer = None
    