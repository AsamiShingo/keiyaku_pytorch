import os
import threading
from keiyakumodel import KeiyakuModel
from transformersbase import TransformersBase, TransformersTokenizerBase
from transformersbert import TransformersBert, TransformersTokenizerBert

class KeiyakuModelFactory:
    MODEL_NAME_BERT="bert"

    MODEL_FULL_NAME_BERT=r"colorfulscoop/bert-base-ja"

    DEFAULT_MODEL_NAME=MODEL_NAME_BERT
    now_model_name:str = ""

    seq_len = 256

    model: TransformersBase = None
    tokenizer: TransformersTokenizerBase = None
    keiyakumodel: KeiyakuModel = None
    model_full_name: str = ""

    craete_transformers_mutex = threading.Lock()

    @classmethod
    def get_keiyakumodel(cls, model_name=DEFAULT_MODEL_NAME, loadweight=True, download=False):
        if model_name != cls.now_model_name:
            cls.get_transfomers(model_name, download)

            cls.keiyakumodel = KeiyakuModel(cls.model, cls.tokenizer)
            if loadweight == True:
                weight_path = os.path.join(os.path.dirname(__file__), r"data", r"model", cls.model.model_name, r"weights.dat")
                cls.keiyakumodel.load_weight(weight_path)

            cls.now_model_name = model_name

        return cls.keiyakumodel, cls.model, cls.tokenizer

    @classmethod
    def get_transfomers(cls, model_name=DEFAULT_MODEL_NAME, download=False):
        cls._craete_transformers(model_name)

        modeldata_path = modeldata_path = os.path.join(os.path.dirname(__file__), r"data", r"model")
        if download == True:
            cls.model.download_save(modeldata_path)
            cls.tokenizer.download_save(modeldata_path)

        cls.model.init_model(modeldata_path)
        cls.tokenizer.init_tokenizer(modeldata_path)

        return cls.model, cls.tokenizer

    @classmethod
    def _craete_transformers(cls, model_name) -> None:
        cls.craete_transformers_mutex.acquire()
        cls.model = None
        cls.tokenizer = None

        if model_name == cls.MODEL_NAME_BERT:
            cls.model = TransformersBert(seq_len=cls.seq_len)
            cls.tokenizer = TransformersTokenizerBert()
            cls.model_full_name = cls.MODEL_FULL_NAME_BERT
        else:
            raise NotImplementedError("model_name error(model_name={})".format(model_name))

        cls.now_model_name = ""
        cls.craete_transformers_mutex.release()


