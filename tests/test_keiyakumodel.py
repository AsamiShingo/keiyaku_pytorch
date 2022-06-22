import pytest
from keiyakumodel import KeiyakuModel
import os
import shutil
from transformersbase import TransformersBase, TransformersTokenizerBase

class TestKeiyakuModel:
    @pytest.fixture(scope="class")
    def tmpsave_dir(self, tmpdir_factory):
        tmpdir_path = tmpdir_factory.mktemp("TestKeiyakuModel")

        yield tmpdir_path

        shutil.rmtree(tmpdir_path)

    def test_init_model(self, test_transformers_empty: TransformersBase, test_transformers_tokenizer_empty: TransformersTokenizerBase):
        keiyaku_model = KeiyakuModel(test_transformers_empty, test_transformers_tokenizer_empty, 10)
        assert keiyaku_model.seq_len == test_transformers_empty.seq_len
        assert keiyaku_model.output_class1_num == 10
 