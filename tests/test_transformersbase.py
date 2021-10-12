import pytest
import transformers
from keras_bert.backend import backend as K
from transformersbase import TransformersBase, TransformersTokenizerBase

class TestTransformersBase:
    def test_get_inputs(self, test_transformers_empty: TransformersBase):
        inputs = test_transformers_empty.get_inputs()

        assert len(inputs) == 3
        assert K.int_shape(inputs[0]) == (None, test_transformers_empty.seq_len)
        assert K.int_shape(inputs[1]) == (None, test_transformers_empty.seq_len)
        assert K.int_shape(inputs[2]) == (None, test_transformers_empty.seq_len)

    def test_get_transformers_model(self, test_transformers_empty: TransformersBase):
        assert test_transformers_empty.get_transformers_model() == None

    def test_get_transformers_output(self, test_transformers_empty: TransformersBase):
        output = test_transformers_empty.get_transformers_output()
        assert K.int_shape(output) == (None, test_transformers_empty.seq_len)
    
    def test_set_trainable(self, test_transformers_empty: TransformersBase, mocker):
        mock = mocker.MagicMock()
        test_transformers_empty.transformers_model = mock
        test_transformers_empty.set_trainable(False)
        assert mock.trainable == False
        test_transformers_empty.set_trainable(True)
        assert mock.trainable == True

class TestTransformersTokenizerBase:

    def test_get_index(self, test_transformers_tokenizer_empty: TransformersTokenizerBase, mocker):
        def mock_encode(text, add_special_tokens):
            if text == "イギリスグループ運動":
                return [ 6302, 488, 542 ]
            elif text == "イギリス":
                return [6302]
            else:
                return [0]

        mock = mocker.Mock(spec=transformers.PreTrainedTokenizerBase)
        mock.encode = mocker.Mock(side_effect = mock_encode)     
        test_transformers_tokenizer_empty.tokenizer = mock

        idx = test_transformers_tokenizer_empty.get_index('イギリス')
        assert idx == 6302

        idxes = test_transformers_tokenizer_empty.get_indexes('イギリスグループ運動')
        assert idxes == [6302, 488, 542]

    def test_get_vocab(self, test_transformers_tokenizer_empty: TransformersTokenizerBase, mocker):
        def mock_convert_ids_to_tokens(ids, skip_special_tokens):
            result = []
            for id in ids:
                if id == 6302:
                    result.append("イギリス")
                elif id == 488:
                    result.append("グループ")
                elif id == 542:
                    result.append("運動")
                else:
                    result.append("")

            return result
        
        mock = mocker.Mock(spec=transformers.PreTrainedTokenizerBase)
        mock.convert_ids_to_tokens = mocker.Mock(side_effect = mock_convert_ids_to_tokens)        
        test_transformers_tokenizer_empty.tokenizer = mock
                    
        vocab = test_transformers_tokenizer_empty.get_vocab(6302)
        assert vocab == "イギリス"
        vocab = test_transformers_tokenizer_empty.get_vocab(488)
        assert vocab == "グループ"
        vocab = test_transformers_tokenizer_empty.get_vocab(542)
        assert vocab == "運動"

        vocabs = test_transformers_tokenizer_empty.get_vocabs([6302, 488, 542])
        assert vocabs == ["イギリス", "グループ", "運動"]

    def test_get_special_idx(self, test_transformers_tokenizer_empty: TransformersTokenizerBase, mocker):
        mock = mocker.Mock()
        mock.pad_token_id = 1
        mock.unk_token_id = 2
        mock.cls_token_id = 3
        mock.sep_token_id = 4
        test_transformers_tokenizer_empty.tokenizer = mock

        assert test_transformers_tokenizer_empty.get_pad_idx() == 1
        assert test_transformers_tokenizer_empty.get_unk_idx() == 2
        assert test_transformers_tokenizer_empty.get_cls_idx() == 3
        assert test_transformers_tokenizer_empty.get_sep_idx() == 4