from abc import ABC, abstractmethod
from typing import List, Any
import tensorflow as tf
import numpy as np
import transformers
import re
from transformersbase import TransformersBase, TransformersTokenizerBase

class TransformersRoberta(TransformersBase):
    def __init__(self, model_name="roberta-rinna", seq_len=256):
        super().__init__(model_name, seq_len)
        
    def init_model(self, model_dir_path: str) -> None:
        input_ids = tf.keras.layers.Input((self.seq_len,), dtype=tf.int32)
        input_attention_mask = tf.keras.layers.Input((self.seq_len,), dtype=tf.int32)

        model_path = self._get_model_path(model_dir_path)

        self.inputs = [ input_ids, input_attention_mask ]
        self.transformers_model = transformers.TFRobertaModel.from_pretrained(model_path, local_files_only=True)
        self.outputs = self.transformers_model(self.inputs)

    def download_save(self, model_full_name: str, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        model = transformers.TFRobertaModel.from_pretrained(model_full_name)
        model.save_pretrained(model_path)

class TransformersTokenizerRoberta(TransformersTokenizerBase):
    def __init__(self, model_name="roberta-rinna"):
        super().__init__(model_name)

    def get_sep_idx(self) -> int:
        return self.tokenizer.eos_token_id

    def convert_vocabs(self, vocabs: List[str]) -> List[str]:
        return [ re.sub(r"^▁", "", vocab) for vocab in vocabs ]

    def init_tokenizer(self, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(model_path, local_files_only=True)

    def download_save(self, model_full_name: str, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_full_name)
        tokenizer.save_pretrained(model_path)

    def get_keiyaku_indexes(self, text1: str, text2: str, max_seq_len: int) -> List[int]:
        input_ids = super().get_keiyaku_indexes(text1, text2, max_seq_len)
        return input_ids[1:]

    def keiyaku_encode(self, ids: List[int], seq_len: int) -> Any:
        encode = super().keiyaku_encode(ids, seq_len)
        return encode[0:2]

