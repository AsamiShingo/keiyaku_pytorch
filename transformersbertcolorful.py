from abc import ABC, abstractmethod
from typing import List, Any
import tensorflow as tf
import numpy as np
import transformers
import re
from transformersbase import TransformersBase, TransformersTokenizerBase

class TransformersBertColorful(TransformersBase):
    def __init__(self, model_name="bert-colorfulscoop", seq_len=256):
        super().__init__(model_name, seq_len)
        
    def init_model(self, model_dir_path: str) -> None:
        input_ids = tf.keras.layers.Input((self.seq_len,), dtype=tf.int32)
        input_attention_mask = tf.keras.layers.Input((self.seq_len,), dtype=tf.int32)
        input_token_type = tf.keras.layers.Input((self.seq_len,), dtype=tf.int32)

        model_path = self._get_model_path(model_dir_path)

        self.inputs = [ input_ids, input_attention_mask, input_token_type ]
        self.transformers_model = transformers.TFBertModel.from_pretrained(model_path, local_files_only=True)
        self.outputs = self.transformers_model(self.inputs)

    def download_save(self, model_full_name: str, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        model = transformers.TFBertModel.from_pretrained(model_full_name)
        model.save_pretrained(model_path)

class TransformersTokenizerBertColorful(TransformersTokenizerBase):
    def __init__(self, model_name="bert-colorfulscoop"):
        super().__init__(model_name)

    def init_tokenizer(self, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        self.tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(model_path, local_files_only=True)

    def download_save(self, model_full_name: str, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(model_full_name)
        tokenizer.save_pretrained(model_path)

