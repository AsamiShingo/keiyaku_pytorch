import pytest
import glob
import os
from transformersfactory import TransformersFactory
from transformersbert import TransformersBert, TransformersTokenizerBert
from transformersbertcolorful import TransformersBertColorful, TransformersTokenizerBertColorful
from transformersroberta import TransformersRoberta, TransformersTokenizerRoberta

class TestTransformersFactory:

    def test_get_transfomers(self):

        model, tokenizer = TransformersFactory.get_transfomers()
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        model, tokenizer = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_BERT)
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        model, tokenizer = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_BERTCOLORFUL)
        assert type(model) == TransformersBertColorful
        assert type(tokenizer) == TransformersTokenizerBertColorful

        model, tokenizer = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_ROBERTA)
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        model2, tokenizer2 = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_ROBERTA)
        assert model == model2
        assert tokenizer == tokenizer2

        with pytest.raises(NotImplementedError):
            _, _ = TransformersFactory.get_transfomers("ERROR")

    @pytest.mark.skip(reason='not testdata update')
    def test_download_transformers(self):

        model, tokenizer = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_BERT, True)
        assert type(model) == TransformersBert
        assert type(tokenizer) == TransformersTokenizerBert

        model, tokenizer = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_BERTCOLORFUL, True)
        assert type(model) == TransformersBertColorful
        assert type(tokenizer) == TransformersTokenizerBertColorful

        model, tokenizer = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_ROBERTA, True)
        assert type(model) == TransformersRoberta
        assert type(tokenizer) == TransformersTokenizerRoberta

        model2, tokenizer2 = TransformersFactory.get_transfomers(TransformersFactory.MODEL_NAME_ROBERTA, True)
        assert model != model2
        assert tokenizer != tokenizer2
        
