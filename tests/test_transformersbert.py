import pytest
from keras_bert.backend import backend as K
from transformersbert import TransformersBert, TransformersTokenizer

class TestTransformersBert:

    def test_get_output_layer(self, test_transformers_bert: TransformersBert):
        test_transformers_bert.init_bert_model(20)

        output = test_transformers_bert.get_output_layer()
        assert K.int_shape(output) == (None, 768)

    def test_get_inputs_base(self, test_transformers_bert: TransformersBert):
        inputs = test_transformers_bert.get_inputs_base()

        assert len(inputs) == 2
        assert K.int_shape(inputs[0]) == (None, 20)
        assert K.int_shape(inputs[1]) == (None, 20)
    
    def test_set_trainable(self, test_transformers_bert: TransformersBert):
        test_transformers_bert.set_trainable(False)
        for layer in test_transformers_bert.get_bert_model().layers:
            assert layer.trainable == False
            
        test_transformers_bert.set_trainable(True)
        for layer in test_transformers_bert.get_bert_model().layers:
            assert layer.trainable == True

class TestTransformersTokenizer:
    """
    get_index
    """
    def test_get_index(self, test_transformers_tokenizer: TransformersTokenizer):
        idx = test_transformers_tokenizer.get_index('から')
        assert idx == 28

    """
    get_indexes
    """
    def test_get_indexes(self, test_transformers_tokenizer: TransformersTokenizer):
        idx = test_transformers_tokenizer.get_indexes('イギリスグループ運動')
        assert idx == [6302, 488, 542]

    """
    get_vocab
    """
    def test_get_vocab(self, test_transformers_tokenizer: TransformersTokenizer):
        vocab = test_transformers_tokenizer.get_vocab(6302)
        assert vocab == "イギリス"
        vocab = test_transformers_tokenizer.get_vocab(488)
        assert vocab == "グループ"
        vocab = test_transformers_tokenizer.get_vocab(542)
        assert vocab == "運動"

    """
    get_vocabs
    指定したIndexに対して、sp.DecodePiecesの結果が取得できること
    """
    def test_get_vocabs(self, test_transformers_tokenizer: TransformersTokenizer):
        vocabs = test_transformers_tokenizer.get_vocabs([6302, 488, 542])
        assert vocabs == ["イギリス", "グループ", "運動"]

    """
    get_pad_idx, get_cls_idx, get_sep_idx, get_unk_idx
    特殊IDを取得できること
    """
    def test_get_special_idx(self, test_transformers_tokenizer: TransformersTokenizer):
        assert test_transformers_tokenizer.get_pad_idx() == 3
        assert test_transformers_tokenizer.get_unk_idx() == 0
        assert test_transformers_tokenizer.get_cls_idx() == 4
        assert test_transformers_tokenizer.get_sep_idx() == 2

    def test_encode(self, test_transformers_tokenizer: TransformersTokenizer):
        sep_idx = test_transformers_tokenizer.get_sep_idx()

        encode = test_transformers_tokenizer.encode('イギリスグループ運動')
        assert len(encode) == 2
        assert encode[0] == [6302, 488, 542, sep_idx]
        assert encode[1] == [1, 1, 1, 1]

        encode = test_transformers_tokenizer.encode('イギリスグループ運動', 'イギリスグループ運動')
        assert len(encode) == 2
        assert encode[0] == [6302, 488, 542, sep_idx, 6302, 488, 542, sep_idx]
        assert encode[1] == [1, 1, 1, 1, 1, 1, 1, 1]





