import pytest
import transformers
import os
from transformersbase import TransformersBase, TransformersTokenizerBase

class TestTransformersBase:
    def test_get_transformers_model(self, test_transformers_empty: TransformersBase):
        assert test_transformers_empty.get_transformers_model() == None
        
    def test_get_model_path(self, test_transformers_empty: TransformersBase):
        resultpath = os.path.join(r"D:\TEST", test_transformers_empty.model_name)
        assert test_transformers_empty._get_model_path(r"D:\TEST") == resultpath

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
        
    def test_get_keiyaku_indexes(self, test_transformers_tokenizer_empty: TransformersTokenizerBase, mocker):
        def mock_encode(text, add_special_tokens):
            if text == "A":
                return [ 10, 11, 12, 13, 14 ]
            elif text == "B":
                return [ 15, 16, 17, 18, 19 ]
            else:
                return [0]
            

        mock = mocker.Mock(spec=transformers.PreTrainedTokenizerBase)
        mock.encode = mocker.Mock(side_effect = mock_encode)
        mock.pad_token_id = 1
        mock.unk_token_id = 2
        mock.cls_token_id = 3
        mock.sep_token_id = 4
        test_transformers_tokenizer_empty.tokenizer = mock

        indexes = test_transformers_tokenizer_empty.get_keiyaku_indexes("A", "B", 20)
        assert indexes == [ 3, 10, 11, 12, 13, 14, 4, 15, 16, 17, 18, 19, 4 ]
        
        indexes = test_transformers_tokenizer_empty.get_keiyaku_indexes("A", "B", 7)
        assert indexes == [ 3, 10, 14, 4, 15, 19, 4 ]
        
    def test_keiyaku_encode(self, test_transformers_tokenizer_empty: TransformersTokenizerBase, mocker):
        mock = mocker.Mock(spec=transformers.PreTrainedTokenizerBase)
        mock.pad_token_id = 1
        mock.unk_token_id = 2
        mock.cls_token_id = 3
        mock.sep_token_id = 4
        test_transformers_tokenizer_empty.tokenizer = mock

        indexes = [ 3, 10, 11, 12, 13, 14, 4, 15, 16, 17, 18, 19, 4 ]
         
        
        result = test_transformers_tokenizer_empty.keiyaku_encode(indexes, 15)
        assert result[0] == [ 3, 10, 11, 12, 13, 14,  4, 15, 16, 17, 18, 19,  4,  1,  1 ]
        assert result[1] == [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0 ]
        assert result[2] == [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1 ]
        
    def test_encode(self, test_transformers_tokenizer_empty: TransformersTokenizerBase, mocker):
        mock = mocker.Mock(spec=transformers.PreTrainedTokenizerBase)
        mock.return_value = { "input_ids":[1, 2, 3], "attention_mask":[4, 5, 6], "token_type_ids":[7, 8, 9]}
        test_transformers_tokenizer_empty.tokenizer = mock

        indexes = test_transformers_tokenizer_empty.encode("A", "B")
        assert indexes[0] == [ 1, 2, 3 ]
        assert indexes[1] == [ 4, 5, 6 ]
        assert indexes[2] == [ 7, 8, 9 ]
        
    def test_get_model_path(self, test_transformers_tokenizer_empty: TransformersTokenizerBase):
        resultpath = os.path.join(r"D:\TEST", test_transformers_tokenizer_empty.model_name)
        assert test_transformers_tokenizer_empty._get_model_path(r"D:\TEST") == resultpath