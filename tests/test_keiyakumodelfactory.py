import pytest
import glob
import os
from keiyakumodel import KeiyakuModel
from keiyakumodelfactory import KeiyakuModelFactory
from transformersbert import TransformersBert, TransformersTokenizerBert
from transformersbertcolorful import TransformersBertColorful, TransformersTokenizerBertColorful
from transformersroberta import TransformersRoberta, TransformersTokenizerRoberta

class TestKeiyakuModelFactory:

    def test_get_keiyakumodel(self):
        keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel()
        assert type(keiyakumodel) == KeiyakuModel
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_BERT, False)
        assert type(keiyakumodel) == KeiyakuModel
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_BERTCOLORFUL, False)
        assert type(keiyakumodel) == KeiyakuModel
        assert type(model) == TransformersBertColorful
        assert type(tokenizer) == TransformersTokenizerBertColorful

        keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_ROBERTA, False)
        assert type(keiyakumodel) == KeiyakuModel
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        keiyakumodel, model2, tokenizer2 = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_ROBERTA, False)
        assert keiyakumodel == keiyakumodel
        assert model == model2
        assert tokenizer == tokenizer2

        keiyakumode2, model2, tokenizer2 = KeiyakuModelFactory.get_keiyakumodel(KeiyakuModelFactory.MODEL_NAME_ROBERTA, True)
        assert keiyakumodel == keiyakumode2
        assert model == model2
        assert tokenizer == tokenizer2

        with pytest.raises(NotImplementedError):
            _, _, _ = KeiyakuModelFactory.get_keiyakumodel("ERROR")

    def test_get_transfomers(self):

        model, tokenizer = KeiyakuModelFactory.get_transfomers()
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_BERT)
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_BERTCOLORFUL)
        assert type(model) == TransformersBertColorful
        assert type(tokenizer) == TransformersTokenizerBertColorful

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_ROBERTA)
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        model2, tokenizer2 = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_ROBERTA)
        assert model != model2
        assert tokenizer != tokenizer2

        with pytest.raises(NotImplementedError):
            _, _ = KeiyakuModelFactory.get_transfomers("ERROR")

    @pytest.mark.skip(reason='not testdata update')
    def test_download_transformers(self):

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_BERT, True)
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_BERTCOLORFUL, True)
        assert type(model) == TransformersBertColorful
        assert type(tokenizer) == TransformersTokenizerBertColorful

        model, tokenizer = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_ROBERTA, True)
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        model2, tokenizer2 = KeiyakuModelFactory.get_transfomers(KeiyakuModelFactory.MODEL_NAME_ROBERTA, True)
        assert model != model2
        assert tokenizer != tokenizer2
        
