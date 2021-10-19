from keiyakudata import KeiyakuData
from keiyakumodel import KeiyakuModel
import transformersbert
import transformersbertcolorful
import transformersroberta
import os
import datetime
import sys

starttime=datetime.datetime.now().strftime('%Y%m%d%H%M%S')

keiyakudata_path = r".\data\keiyakudata.csv"
modeldata_path = r".\data\model"
save_dir = r".\savedir"
epoch_num = 20

model_name = "roberta"
if len(sys.argv) >= 2:
    model_name = sys.argv[1]

if model_name == "bert":
    model = transformersbert.TransformersBert()
    tokenizer = transformersbert.TransformersTokenizerBert()
elif model_name == "bertcolorful":
    model = transformersbertcolorful.TransformersBertColorful()
    tokenizer = transformersbertcolorful.TransformersTokenizerBertColorful()
elif model_name == "roberta":
    model = transformersroberta.TransformersRoberta()
    tokenizer = transformersroberta.TransformersTokenizerRoberta()
else:
    print("model_name error(model_name={})".format(model_name))
    sys.exit(9)
    
keiyakumodel = KeiyakuModel(tokenizer)

model.init_model(modeldata_path)
tokenizer.init_tokenizer(modeldata_path)
keiyakumodel.init_model(model)

keiyakudata = KeiyakuData(keiyakudata_path)
datas = keiyakudata.get_study_group_datas(tokenizer, model.seq_len)

save_dir = os.path.join(save_dir, starttime + "_" + model.model_name)
keiyakumodel.train_model(datas, epoch_num, save_dir)

