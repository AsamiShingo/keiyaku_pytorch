from keiyakudata import KeiyakuDataset, KeiyakuDataLoader
from keiyakuai import KeiyakuAI
from keiyakumodelfactory import KeiyakuModelFactory
import numpy as np
import sys
import pandas as pd

keiyakudata_path = r".\data\keiyaku_predict.csv"
save_dir = r".\savedir"
score_threshold = 0.5

model_name = KeiyakuModelFactory.DEFAULT_MODEL_NAME
if len(sys.argv) >= 2:
    model_name = sys.argv[1]

keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel(model_name)

predict_data = KeiyakuDataset(keiyakudata_path, True, 256, 6, tokenizer)
predict_loader = KeiyakuDataLoader(predict_data, False, 20)

ai = KeiyakuAI(keiyakumodel)
predict_results = ai.predict(predict_loader)

np.set_printoptions(precision=2, floatmode='fixed')

bef_file = ""
for datas, score1, score2 in zip(predict_data.get_datas(), predict_results[0], predict_results[1]):
    file = datas[0]
    sentense =  datas[6] if not pd.isnull(datas[6]) else ""
    kind1 = score2.argmax()

    if bef_file != file:
        print("{}*********************************************".format(file))

    if score1 >= score_threshold:
        print("{}---------------------------------------------".format(score1))
        
    print("{}-{:0.2f}:{}".format(kind1, score2[kind1], sentense))
    
    bef_file = file


    


