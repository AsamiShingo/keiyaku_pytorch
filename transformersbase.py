from abc import ABC, abstractmethod
from typing import List, Dict, Any
import tensorflow as tf
import transformers
import os

class TransformersBase(ABC):
    def __init__(self, model_name: str, seq_len: int):
        self.model_name = model_name
        self.seq_len = seq_len

        self.inputs = None
        self.outputs = None        
        self.transformers_model: transformers.TFPreTrainedModel = None
        
    def get_transformers_model(self) -> transformers.TFPreTrainedModel:
        return self.transformers_model

    def get_inputs(self) -> List[tf.keras.layers.Layer]:
        return self.inputs

    def get_transformers_output(self) -> tf.keras.layers.Layer:
        return self.outputs["pooler_output"]

    def set_trainable(self, training: bool) -> None:
        self.transformers_model.trainable = training

    @abstractmethod
    def init_model(self, model_dir_path: str) -> None:
        pass

    @abstractmethod
    def download_save(self, model_full_name: str, model_dir_path: str) -> None:
        pass

    def _get_model_path(self, model_dir_path: str) -> str:
        return os.path.join(model_dir_path, self.model_name)         

class TransformersTokenizerBase(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.tokenizer: transformers.PreTrainedTokenizerBase = None

    def get_index(self, vocab: str) -> int:
        return self.get_indexes(vocab)[0]

    def get_indexes(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_vocab(self, index: int) -> str:
        return self.get_vocabs([index])[0]
        
    def get_vocabs(self, indexes: List[int]) -> List[str]:
        vocabs = self.tokenizer.convert_ids_to_tokens(indexes, skip_special_tokens=True)
        return self._convert_vocabs(vocabs)

    def get_cls_idx(self) -> int:
        return self.tokenizer.cls_token_id

    def get_sep_idx(self) -> int:
        return self.tokenizer.sep_token_id

    def get_pad_idx(self) -> int:
        return self.tokenizer.pad_token_id

    def get_unk_idx(self) -> int:
        return self.tokenizer.unk_token_id

    def encode(self, text1: str, text2: str = None) -> List[List[int]]:
        encode = self.tokenizer(text1, text2)
        result = []
        if "input_ids" in encode:
            result.append(encode['input_ids'])
        if "attention_mask" in encode:
            result.append(encode['attention_mask'])
        if "token_type_ids" in encode:
            result.append(encode['token_type_ids'])

        # result.append(encode['input_ids'])
        # result.append(encode['attention_mask'])
        # result.append(encode['token_type_ids'])

        return result

    def get_keiyaku_indexes(self, text1: str, text2: str, max_seq_len: int) -> List[int]:
        cls_idx = self.get_cls_idx()
        sep_idx = self.get_sep_idx()
        cut_len = int((max_seq_len - 3)/2)
        cut_len_herf = int(cut_len / 2)
        
        now_text_idx = self.get_indexes(text1)
        bef_text_idx = self.get_indexes(text2)

        if len(now_text_idx) > cut_len:
            now_text_idx = now_text_idx[:cut_len_herf] + now_text_idx[-cut_len_herf:]

        if len(bef_text_idx) > cut_len:
            bef_text_idx = bef_text_idx[:cut_len_herf] + bef_text_idx[-cut_len_herf:]

        return [cls_idx] + now_text_idx + [sep_idx] + bef_text_idx + [sep_idx]

    def keiyaku_encode(self, ids: List[int], seq_len: int) -> Any:
        sep_idx = self.get_sep_idx()
        pad_idx = self.get_pad_idx()

        input_ids = ids + ([pad_idx] * (seq_len - len(ids)))
        input_attention_mask = [ 1 if id != pad_idx else 0 for id in input_ids ]
        token_type_ids = [1] * len(input_ids)
        for i, id in enumerate(input_ids):
            token_type_ids[i] = 0
            if id == sep_idx:
                break

        return [input_ids, input_attention_mask, token_type_ids]

    @abstractmethod
    def init_tokenizer(self, model_dir_path: str) -> None:
        pass

    @abstractmethod
    def download_save(cls, model_full_name: str, save_dir: str) -> None:
        pass    

    def _convert_vocabs(self, vocabs: List[str]) -> List[str]:
        return vocabs
        
    def _get_model_path(self, model_dir_path: str) -> str:
        return os.path.join(model_dir_path, self.model_name)
