from keiyakudata import KeiyakuData
from keiyakumodel import KeiyakuModel
from transfomersbert import TransfomersBert, TransfomersTokenizer
import numpy as np

keiyakudata_path = r".\data\keiyakudata.csv"
keiyakuweight_path = r".\data\transfomers_weights"
score_threshold = 0.5
seq_len = 256

tokenizer = TransfomersTokenizer()
bert = TransfomersBert()
bert.init_bert_model(seq_len)

model = KeiyakuModel(tokenizer)
model.init_model(bert)
model.load_weight(keiyakuweight_path)

keiyakudata = KeiyakuData(keiyakudata_path)
datas = keiyakudata.get_datas()
predict_datas = keiyakudata.get_group_datas(tokenizer, model.seq_len)

np.set_printoptions(precision=2, floatmode='fixed')

bef_file = ""
for i in range(0, len(datas), 1000):
    targets = datas[i:i+1000]
    predict_targets = predict_datas[i:i+1000]
    scores = model.predict(predict_targets)

    for target, score1, score2 in zip(targets, scores[0], scores[1]):
        file = target[0]
        sentense = target[6]
        kind1 = score2.argmax()

        if bef_file != file:
            print("{}*********************************************".format(file))

        if score1 >= score_threshold:
            print("{}---------------------------------------------".format(score1))
            
        print("{}-{:0.2f}:{}".format(kind1, score2[kind1], sentense))
        
        bef_file = file


    


