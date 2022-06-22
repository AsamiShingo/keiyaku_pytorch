import pytest
import glob
import os
from keiyakumodel import KeiyakuModel
from keiyakumodelfactory import KeiyakuModelFactory
from transformersbert import TransformersBert, TransformersTokenizerBert

class TestKeiyakuModelFactory:

    def test_get_keiyakumodel(self):
        keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel()
        assert type(keiyakumodel) == KeiyakuModel
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_BERT, False)
        assert type(keiyakumodel) == KeiyakuModel
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        keiyakumodel2, model2, tokenizer2 = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_BERT, False)
        assert keiyakumodel == keiyakumodel2
        assert model == model2
        assert tokenizer == tokenizer2

        keiyakumode3, model3, tokenizer3 = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_BERT, True)
        assert keiyakumodel == keiyakumode3
        assert model == model3
        assert tokenizer == tokenizer3

        with pytest.raises(NotImplementedError):
            _, _, _ = KeiyakuModelFactory.get_keiyakumodel("ERROR")

    def test_get_transfomers(self):

        model, tokenizer = KeiyakuModelFactory.get_transfomers()
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_BERT)
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert
        
        with pytest.raises(NotImplementedError):
            _, _ = KeiyakuModelFactory.get_transfomers("ERROR")

    @pytest.mark.skip(reason='not testdata update')
    def test_download_transformers(self):

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_BERT, True)
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert
        
