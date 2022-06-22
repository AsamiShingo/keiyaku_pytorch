from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
import transformers
import re
from transformersbase import TransformersBase, TransformersTokenizerBase

class TransformersBert(TransformersBase):
    MODEL_NAME="bert"
    MODEL_FULL_NAME="colorfulscoop/bert-base-ja"
    
    def __init__(self, seq_len=256, output_dim=768):
        super().__init__(self.MODEL_NAME, self.MODEL_FULL_NAME, seq_len, output_dim)
        
    def init_model(self, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        self.transformers_model = transformers.BertModel.from_pretrained(model_path, local_files_only=True)
        
    def download_save(self, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        model = transformers.BertModel.from_pretrained(self.model_full_name)
        model.save_pretrained(model_path)

class TransformersTokenizerBert(TransformersTokenizerBase):
    MODEL_NAME="bert"
    MODEL_FULL_NAME="colorfulscoop/bert-base-ja"
    
    def __init__(self):
        super().__init__(self.MODEL_NAME, self.MODEL_FULL_NAME)

    def _convert_vocabs(self, vocabs: List[str]) -> List[str]:
        return [ re.sub(r"^##", "", vocab) for vocab in vocabs ]

    def init_tokenizer(self, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        self.tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(model_path, local_files_only=True)

    def download_save(self, model_dir_path: str) -> None:
        model_path = self._get_model_path(model_dir_path)
        tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(self.model_full_name)
        tokenizer.save_pretrained(model_path)

