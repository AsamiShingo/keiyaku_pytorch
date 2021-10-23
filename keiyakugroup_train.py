from keiyakudata import KeiyakuData
from keiyakumodel import KeiyakuModel
from keiyakumodelfactory import KeiyakuModelFactory
import os
import datetime
import sys

starttime=datetime.datetime.now().strftime('%Y%m%d%H%M%S')

keiyakudata_path = r".\data\keiyakudata.csv"
save_dir = r".\savedir"
epoch_num = 20

model_name = KeiyakuModelFactory.DEFAULT_MODEL_NAME
if len(sys.argv) >= 2:
    model_name = sys.argv[1]

keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel(model_name, loadweight=False)

keiyakudata = KeiyakuData(keiyakudata_path)
datas = keiyakudata.get_study_group_datas(tokenizer, model.seq_len)

save_dir = os.path.join(save_dir, starttime + "_" + model.model_name)
keiyakumodel.train_model(datas, epoch_num, save_dir)

