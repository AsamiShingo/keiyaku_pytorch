import pytest
import tensorflow as tf
from keras_bert.backend import backend as K
from keiyakumodel import KeiyakuModel
from keiyakudata import KeiyakuData
import os
import shutil
import glob
import numpy as np
import pandas as pd
from transformersbert import TransformersBert, TransformersTokenizer

class TestKeiyakuModel:
    @pytest.fixture(scope="class")
    def tmpsave_dir(self, tmpdir_factory):
        tmpdir_path = tmpdir_factory.mktemp("TestKeiyakuModel")

        yield tmpdir_path

        shutil.rmtree(tmpdir_path)

    def test_init_model(self, test_transformers_bert: TransformersBert, test_transformers_tokenizer: TransformersTokenizer):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer, 10, 11)
        keiyaku_model.init_model(test_transformers_bert)
        
        assert keiyaku_model.seq_len == 20
        assert keiyaku_model.output_class1_num == 10
        assert keiyaku_model.output_class2_num == 11

        assert len(keiyaku_model.model.inputs) == 2
        assert K.int_shape(keiyaku_model.model.inputs[0]) == (None, keiyaku_model.seq_len)
        assert K.int_shape(keiyaku_model.model.inputs[1]) == (None, keiyaku_model.seq_len)
        assert len(keiyaku_model.model.outputs) == 2
        assert K.int_shape(keiyaku_model.model.outputs[0]) == (None, 1)
        assert K.int_shape(keiyaku_model.model.outputs[1]) == (None, 10)

    def test_generate_data(self, test_transformers_bert: TransformersBert, test_transformers_tokenizer: TransformersTokenizer):
        sepidx = test_transformers_tokenizer.get_sep_idx()
        datas = [([10, 11, 12, 13], 0, 3, 2), ([11, 11, sepidx, 13], 1, 4, 3), ([2, 1, 2, 3], 2, 5, 4)]

        keiyaku_model = KeiyakuModel(test_transformers_tokenizer)
        keiyaku_model.init_model(test_transformers_bert)

        loop_num = 0
        for x, y in keiyaku_model._generator_data(datas, 2):
            assert len(x) == 2
            assert len(y) == 2
            
            x1 = x[0]
            x2 = x[1]
            y1 = y[0]
            y2 = y[1]
            assert len(x1) == 2
            assert len(x2) == 2
            assert len(y1) == 2
            assert len(y2) == 2

            assert x1[0][:5].tolist() == [10, 11, 12, 13, 0]
            assert x1[1][:5].tolist() == [11, 11, sepidx, 13, 0]
            assert x2[0][:5].tolist() == [1, 1, 1, 1, 0]
            assert x2[1][:5].tolist() == [1, 1, 1, 1, 0]
            assert y1[0] == 0
            assert y1[1] == 1
            assert y2[0].tolist() == [0, 0, 0, 1, 0, 0]
            assert y2[1].tolist() == [0, 0, 0, 0, 1, 0]

            loop_num += 1
            if loop_num > 3:
                break

    def test_get_learn_rate(self, test_transformers_tokenizer: TransformersTokenizer):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer)

        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch-1) == keiyaku_model.learn_rate_init
        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch) == keiyaku_model.learn_rate_init * keiyaku_model.learn_rate_percent
        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch*2-1) == keiyaku_model.learn_rate_init * keiyaku_model.learn_rate_percent
        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch*2) == keiyaku_model.learn_rate_init * keiyaku_model.learn_rate_percent * keiyaku_model.learn_rate_percent

    def test_get_callbacks(self, test_transformers_tokenizer: TransformersTokenizer, tmpsave_dir):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer)

        callbacks = keiyaku_model._get_callbacks(tmpsave_dir)
        assert len(callbacks) == 3
        assert type(callbacks[0]) is tf.keras.callbacks.ModelCheckpoint
        assert type(callbacks[1]) is KeiyakuModel.ResultOutputCallback
        assert type(callbacks[2]) is tf.keras.callbacks.LearningRateScheduler

    def test_train_model(self, test_keiyakudata: KeiyakuData, test_transformers_bert: TransformersBert, test_transformers_tokenizer: TransformersTokenizer, tmpsave_dir):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer)
        keiyaku_model.init_model(test_transformers_bert)
        
        datas = test_keiyakudata.get_study_group_datas(test_transformers_tokenizer, 10)
        keiyaku_model.train_model(datas, 2, tmpsave_dir)

        assert len(glob.glob(os.path.join(tmpsave_dir, "model_summary.txt"))) == 1
        assert len(glob.glob(os.path.join(tmpsave_dir, "model.json"))) == 1
        assert len(glob.glob(os.path.join(tmpsave_dir, "weights_last-*.data-*"))) == 1
        assert len(glob.glob(os.path.join(tmpsave_dir, "weights_last-*.index"))) == 1
        assert len(glob.glob(os.path.join(tmpsave_dir, "weights_001-*.data-*"))) == 1
        assert len(glob.glob(os.path.join(tmpsave_dir, "weights_001-*.index"))) == 1
        assert len(glob.glob(os.path.join(tmpsave_dir, "result.csv"))) == 1

        df = pd.read_csv(os.path.join(tmpsave_dir, "result.csv"), sep=',')
        assert df.shape == (2, 39)
       
    def test_load_weight(self, mocker, test_transformers_bert: TransformersBert, test_transformers_tokenizer: TransformersTokenizer):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer)
        keiyaku_model.init_model(test_transformers_bert)
        
        load_weights_mock = mocker.spy(keiyaku_model.model, "load_weights")

        test_dir = os.path.join(os.path.dirname(__file__), r"data")    
        keiyaku_model.load_weight(os.path.join(test_dir, "test_transformers_weights"))
        
        load_weights_mock.assert_called_once_with(os.path.join(test_dir, "test_transformers_weights"))
    
    def test_predict(self, test_keiyakudata: KeiyakuData, test_transformers_bert: TransformersBert, test_transformers_tokenizer: TransformersTokenizer):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer)
        keiyaku_model.init_model(test_transformers_bert)
        
        datas = test_keiyakudata.get_group_datas(test_transformers_tokenizer, keiyaku_model.seq_len)
        results = keiyaku_model.predict(datas)
        
        assert len(results[0]) == len(datas)
        assert len(results[1]) == len(datas)
        for result in results[0]:
            assert 0.0 <= result and result <= 1.0
        for result in results[1]:
            assert len(result) == 6

    def test_data_check(self, test_keiyakudata: KeiyakuData, test_transformers_bert: TransformersBert, test_transformers_tokenizer: TransformersTokenizer):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer)
        keiyaku_model.init_model(test_transformers_bert)
        keiyaku_model.seq_len = 102

        datas = test_keiyakudata.get_datas()
        text_bef = datas[7][6]
        text_aft = datas[8][6]

        groups = test_keiyakudata.get_group_datas(test_transformers_tokenizer, 256)
        group = groups[8]

        encoded = test_transformers_tokenizer.encode(text_aft, text_bef)
        result = None
        for x, y in keiyaku_model._generator_data([group], 1):
            result = x
            break
        
        e1 = [ int(data) for data in encoded[0] ] + [0] * (keiyaku_model.seq_len - len(encoded[0]))
        e2 = [ int(data) for data in encoded[1] ] + [0] * (keiyaku_model.seq_len - len(encoded[1]))
        
        r1 = result[0].tolist()[0]
        r2 = result[1].tolist()[0]

        assert e1 == r1
        assert e2 == r2


    