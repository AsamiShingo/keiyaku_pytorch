import pytest
import glob
import os
import pandas as pd
import tensorflow.keras.backend as K
from keiyakumodel import KeiyakuModel
from keiyakudata import KeiyakuData
from transformersbert import TransformersBert, TransformersTokenizerBert

class TestBert:
    @pytest.fixture(scope="class")
    def test_transformers_bert(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp("test_transformers_bert")

        model = TransformersBert(seq_len = 20)
        model.download_save("cl-tohoku/bert-base-japanese-v2", tmpdir)
        model.init_model(tmpdir)    
        yield model
        model = None

    @pytest.fixture(scope="class")
    def test_transformers_tokenizer_bert(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp("test_transformers_tokenizer_bert")

        tokenizer = TransformersTokenizerBert()
        tokenizer.download_save("cl-tohoku/bert-base-japanese-v2", tmpdir)
        tokenizer.init_tokenizer(tmpdir)    
        yield tokenizer
        tokenizer = None

    class TestTransformersBert:

        def test_get_inputs(self, test_transformers_bert: TransformersBert):
            inputs = test_transformers_bert.get_inputs()
            assert len(inputs) == 3
            assert K.int_shape(inputs[0]) == (None, 20)
            assert K.int_shape(inputs[1]) == (None, 20)
            assert K.int_shape(inputs[2]) == (None, 20)

        def test_get_transformers_output(self, test_transformers_bert: TransformersBert):
            output = test_transformers_bert.get_transformers_output()
            assert K.int_shape(output) == (None, 768)

    class TestTransformersTokenizerBert:

        def test_encode(self, test_transformers_tokenizer_bert: TransformersTokenizerBert):
            cls_idx = test_transformers_tokenizer_bert.get_cls_idx()
            sep_idx = test_transformers_tokenizer_bert.get_sep_idx()

            encode = test_transformers_tokenizer_bert.encode('イギリスグループ運動', 'イギリスグループ運動')
            assert len(encode) == 3
            assert encode[0] == [cls_idx, 11491, 17764, 11256, 5244, 1441, sep_idx, 11491, 17764, 11256, 5244, 1441, sep_idx]
            assert encode[1] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            assert encode[2] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        def test_convert_vocabs(self, test_transformers_tokenizer_bert: TransformersTokenizerBert):
            indata = ["##TEST1", "##TEST2", "#TEST3", "TEST4##", "TEST##5"]
            expect = ["TEST1", "TEST2", "#TEST3", "TEST4##", "TEST##5"]
            convert_vocabs = test_transformers_tokenizer_bert._convert_vocabs(indata)
            assert convert_vocabs == expect
            
        def test_get_keiyaku_indexes(self, test_transformers_tokenizer_bert: TransformersTokenizerBert):
            cls_idx = test_transformers_tokenizer_bert.get_cls_idx()
            sep_idx = test_transformers_tokenizer_bert.get_sep_idx()
            encode = test_transformers_tokenizer_bert.encode('イギリスグループ運動', 'イギリスグループ運動')

            keiyaku_indexes = test_transformers_tokenizer_bert.get_keiyaku_indexes('イギリスグループ運動', 'イギリスグループ運動', 100)
            assert keiyaku_indexes == encode[0]
            assert keiyaku_indexes == [cls_idx, 11491, 17764, 11256, 5244, 1441, sep_idx, 11491, 17764, 11256, 5244, 1441, sep_idx]

            keiyaku_indexes = test_transformers_tokenizer_bert.get_keiyaku_indexes('イギリスグループ運動', 'イギリスグループ運動', 7)
            assert keiyaku_indexes == [cls_idx, 11491, 1441, sep_idx, 11491, 1441, sep_idx]

        def test_keiyaku_encode(self, test_transformers_tokenizer_bert: TransformersTokenizerBert):
            cls_idx = test_transformers_tokenizer_bert.get_cls_idx()
            sep_idx = test_transformers_tokenizer_bert.get_sep_idx()
            pad_idx = test_transformers_tokenizer_bert.get_pad_idx()
            
            keiyaku_encode = test_transformers_tokenizer_bert.keiyaku_encode([ cls_idx, 11, 12, 13, 14, sep_idx, 16, 17, 18, 19, sep_idx], 20)
            assert len(keiyaku_encode) == 3
            assert keiyaku_encode[0] == [ cls_idx, 11, 12, 13, 14, sep_idx, 16, 17, 18, 19, sep_idx] + [pad_idx] * 9
            assert keiyaku_encode[1] == [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * 9
            assert keiyaku_encode[2] == [ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] + [1] * 9

            encode = test_transformers_tokenizer_bert.encode('イギリスグループ運動', 'イギリスグループ運動')
            keiyaku_indexes = test_transformers_tokenizer_bert.get_keiyaku_indexes('イギリスグループ運動', 'イギリスグループ運動', 100)
            keiyaku_encode = test_transformers_tokenizer_bert.keiyaku_encode(keiyaku_indexes, len(encode[0]))

            assert keiyaku_encode == encode

        def test_encode_decode(self, test_transformers_tokenizer_bert: TransformersTokenizerBert):
            sentence = "私はこの本(実践機械学習)を読むのに8時間かかった。"
            encode = test_transformers_tokenizer_bert.get_indexes(sentence)
            decode = test_transformers_tokenizer_bert.get_vocabs(encode)
            assert "".join(decode) == sentence

    @pytest.mark.skip(reason='heavy test')
    def test_train_predict(self, test_transformers_bert: TransformersBert, test_transformers_tokenizer_bert: TransformersTokenizerBert, test_keiyakudata: KeiyakuData, tmpdir):
        model = KeiyakuModel(test_transformers_tokenizer_bert)
        model.init_model(test_transformers_bert)
        model.pre_epoch = 1
        
        assert model.seq_len == 20
        assert model.output_class1_num == 6

        study_datas = test_keiyakudata.get_study_group_datas(test_transformers_tokenizer_bert, 20)
        model.train_model(study_datas, 2, tmpdir)

        assert len(glob.glob(os.path.join(tmpdir, "model_summary.txt"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "model.json"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "weights_last-*.data-*"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "weights_last-*.index"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "weights_001-*.data-*"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "weights_001-*.index"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "result_data.csv"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "result_graph1.png"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "result_graph2.png"))) == 1
        assert len(glob.glob(os.path.join(tmpdir, "parameter.json"))) == 1

        df = pd.read_csv(os.path.join(tmpdir, "result_data.csv"), sep=',')
        assert df.shape == (2, 39)

        weight_path = glob.glob(os.path.join(tmpdir, "weights_last-*.data-*"))[0]
        weight_path = weight_path[:weight_path.find(".data-")]

        model = KeiyakuModel(test_transformers_tokenizer_bert)
        model.init_model(test_transformers_bert)
        model.load_weight(weight_path)

        predict_datas = test_keiyakudata.get_group_datas(test_transformers_tokenizer_bert, 20)
        results = model.predict(predict_datas)

        assert len(results[0]) == len(predict_datas)
        assert len(results[1]) == len(predict_datas)
        for result in results[0]:
            assert 0.0 <= result and result <= 1.0
        for result in results[1]:
            assert len(result) == 6

