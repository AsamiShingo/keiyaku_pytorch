from keiyakudata import KeiyakuData
from keiyakumodel import KeiyakuModel
from transformersbert import TransformersBert, TransformersTokenizer
import os
import datetime

starttime=datetime.datetime.now().strftime('%Y%m%d%H%M%S')

keiyakudata_path = r".\data\keiyakudata.csv"
save_dir = r".\savedir"
epoch_num = 20
seq_len = 256

tokenizer = TransformersTokenizer()
bert = TransformersBert()
bert.init_bert_model(seq_len)

model = KeiyakuModel(tokenizer)
model.init_model(bert)
model.learn_rate_init = 0.0001

keiyakudata = KeiyakuData(keiyakudata_path)
datas = keiyakudata.get_study_group_datas(tokenizer, model.seq_len)

save_dir = os.path.join(save_dir, starttime)
model.train_model(datas, epoch_num, save_dir)

