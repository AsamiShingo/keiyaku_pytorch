from logging import setLogRecordFactory
import numpy as np
import tensorflow as tf
import transformers
import re

class TransfomersBert:
    def __init__(self, model_name="rinna/japanese-roberta-base"):
        self.model_name = model_name

        self.input_ids = None
        self.input_attention_mask = None
        # self.input_token_type = None
        self.output = None
        
        self.bert_model = None
        
        self.seq_len = 0
        
    def get_bert_model(self):
        return self.bert_model

    def get_inputs_base(self):
        # return [self.input_ids, self.input_attention_mask, self.input_token_type]
        return [self.input_ids, self.input_attention_mask]

    def get_output_layer(self):
        return self.output["pooler_output"]

    def set_trainable(self, training):
        self.bert_model.trainable = training

    def init_bert_model(self, seq_len):
        self.seq_len = seq_len

        self.input_ids = tf.keras.layers.Input((seq_len,), dtype=tf.int32)
        self.input_attention_mask = tf.keras.layers.Input((seq_len,), dtype=tf.int32)
        # self.input_token_type = tf.keras.layers.Input((seq_len,), dtype=tf.int32)
        
        self.bert_model = transformers.TFRobertaModel.from_pretrained(self.model_name)
        # self.output = self.bert_model([self.input_ids, self.input_attention_mask, self.input_token_type])
        self.output = self.bert_model([self.input_ids, self.input_attention_mask])

class TransfomersTokenizer:
    def __init__(self, model_name="rinna/japanese-roberta-base"):
        self.model_name = model_name
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.model_name)

    def get_index(self, vocab):
        return self.tokenizer.convert_tokens_to_ids(vocab)

    def get_indexes(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_vocab(self, index):
        return self.get_vocabs([index])[0]
        
    def get_vocabs(self, indexes):
        vocabs = self.tokenizer.convert_ids_to_tokens(indexes, skip_special_tokens=True)
        return [ re.sub(r"^▁", "", vocab) for vocab in vocabs ]

    def get_cls_idx(self):
        return self.tokenizer.cls_token_id

    def get_sep_idx(self):
        # return self.tokenizer.sep_token_id
        return self.tokenizer.eos_token_id

    def get_pad_idx(self):
        return self.tokenizer.pad_token_id

    def get_unk_idx(self):
        return self.tokenizer.unk_token_id

    def encode(self, text1, text2=None):
        # input_ids1 = self.get_indexes(text1)
        # input_ids2 = self.get_indexes(text2) if text2 != None else None

        # input_ids = [self.get_cls_idx()] + input_ids1 + [self.get_sep_idx()]
        # if input_ids2 != None:
        #     input_ids = input_ids + input_ids2 + [self.get_sep_idx()]

        # attention_mask = [ 1 for _ in input_ids ]
        # token_type_ids = []
        # type_id = 0
        # for id in input_ids:
        #     token_type_ids.append(type_id)
        #     if id == self.get_sep_idx():
        #         type_id = 1

        encode = self.tokenizer(text1, text2)
        input_ids =encode['input_ids']
        attention_mask = encode['attention_mask']
        # token_type_ids = encode['token_type_ids']

        # token_type_ids = []
        # type_id = 0
        # for id in input_ids:
        #     token_type_ids.append(type_id)
        #     if id == self.get_sep_idx():
        #         type_id = 1

        #return [input_ids, attention_mask, token_type_ids]
        return [input_ids, attention_mask]
