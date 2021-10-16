import pytest
import tensorflow as tf
from keiyakumodel import KeiyakuModel
import tensorflow.keras.backend as K
import os
import shutil
import pandas as pd
from transformersbase import TransformersBase, TransformersTokenizerBase

class TestKeiyakuModel:
    @pytest.fixture(scope="class")
    def tmpsave_dir(self, tmpdir_factory):
        tmpdir_path = tmpdir_factory.mktemp("TestKeiyakuModel")

        yield tmpdir_path

        shutil.rmtree(tmpdir_path)

    def test_init_model(self, test_transformers_empty: TransformersBase, test_transformers_tokenizer_empty: TransformersTokenizerBase):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer_empty, 10)
        keiyaku_model.init_model(test_transformers_empty)
        
        assert keiyaku_model.seq_len == test_transformers_empty.seq_len
        assert keiyaku_model.output_class1_num == 10

        assert len(keiyaku_model.model.inputs) == len(test_transformers_empty.get_inputs())
        for input in keiyaku_model.model.inputs:
            assert K.int_shape(input) == (None, keiyaku_model.seq_len)

        assert len(keiyaku_model.model.outputs) == 2
        assert K.int_shape(keiyaku_model.model.outputs[0]) == (None, 1)
        assert K.int_shape(keiyaku_model.model.outputs[1]) == (None, 10)

    def test_generate_data(self, test_transformers_empty: TransformersBase, test_transformers_tokenizer_empty: TransformersTokenizerBase, mocker):
        
        mocker.patch.object(test_transformers_tokenizer_empty, 'get_pad_idx').return_value = 2
        mocker.patch.object(test_transformers_tokenizer_empty, 'get_sep_idx').return_value = 3
        
        padidx = test_transformers_tokenizer_empty.get_pad_idx()
        sepidx = test_transformers_tokenizer_empty.get_sep_idx()
        datas = [([10, 11, 12, 13], [0, 3, 2]), ([11, 11, sepidx, 13], [1, 4, 3]), ([2, 1, 2, 3], [2, 5, 4])]

        keiyaku_model = KeiyakuModel(test_transformers_tokenizer_empty)
        keiyaku_model.init_model(test_transformers_empty)

        loop_num = 0
        for x, y in keiyaku_model._generator_data(datas, 2):
            assert len(x) == 3
            assert len(y) == 2
            
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            y1 = y[0]
            y2 = y[1]
            assert len(x1) == 2
            assert len(x2) == 2
            assert len(x3) == 2
            assert len(y1) == 2
            assert len(y2) == 2

            assert x1[0][:5].tolist() == [10, 11, 12, 13, padidx]
            assert x1[1][:5].tolist() == [11, 11, sepidx, 13, padidx]
            assert x2[0][:5].tolist() == [1, 1, 1, 1, 0]
            assert x2[1][:5].tolist() == [1, 1, 1, 1, 0]
            assert x3[0][:5].tolist() == [0, 0, 0, 0, 0]
            assert x3[1][:5].tolist() == [0, 0, 0, 1, 1]
            assert y1[0] == 0
            assert y1[1] == 1
            assert y2[0].tolist() == [0, 0, 0, 1, 0, 0]
            assert y2[1].tolist() == [0, 0, 0, 0, 1, 0]

            loop_num += 1
            if loop_num > 3:
                break

    def test_get_learn_rate(self, test_transformers_tokenizer_empty: TransformersTokenizerBase):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer_empty)

        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch-1) == keiyaku_model.learn_rate_init
        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch) == keiyaku_model.learn_rate_init * keiyaku_model.learn_rate_percent
        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch*2-1) == keiyaku_model.learn_rate_init * keiyaku_model.learn_rate_percent
        assert keiyaku_model._get_learn_rate(keiyaku_model.learn_rate_epoch*2) == keiyaku_model.learn_rate_init * keiyaku_model.learn_rate_percent * keiyaku_model.learn_rate_percent

    def test_get_callbacks(self, test_transformers_tokenizer_empty: TransformersTokenizerBase, tmpsave_dir):
        keiyaku_model = KeiyakuModel(test_transformers_tokenizer_empty)

        callbacks = keiyaku_model._get_callbacks(tmpsave_dir)
        assert len(callbacks) == 3
        assert type(callbacks[0]) is tf.keras.callbacks.ModelCheckpoint
        assert type(callbacks[1]) is KeiyakuModel.ResultOutputCallback
        assert type(callbacks[2]) is tf.keras.callbacks.LearningRateScheduler

    def test_callback_create_graph(self, tmpsave_dir):
        df = pd.DataFrame(columns=["epoch", "output1", "output2", "output3", "output4"])
        df = df.append(pd.Series([1, 0.00, 0.25, 0.8, 1], index = ["epoch", "output1", "output2", "output3", "output4"]), ignore_index = True)
        df = df.append(pd.Series([2, 0.25, 0.25, 0.7, 3], index = ["epoch", "output1", "output2", "output3", "output4"]), ignore_index = True)
        df = df.append(pd.Series([3, 0.50, 0.5, 0.6, 2], index = ["epoch", "output1", "output2", "output3", "output4"]), ignore_index = True)
        df = df.append(pd.Series([4, 0.75, 0.5, 0.5, 1], index = ["epoch", "output1", "output2", "output3", "output4"]), ignore_index = True)
        df = df.append(pd.Series([5, 1.00, 0.5, 0.4, 0], index = ["epoch", "output1", "output2", "output3", "output4"]), ignore_index = True)

        filepath = os.path.join(tmpsave_dir, "test.png")
        callback = KeiyakuModel.ResultOutputCallback(tmpsave_dir)
        callback._create_graph("title", df, "epoch", ["output1", "output2", "output3"], ["output4"], filepath)
        assert os.path.exists(filepath) == True
