from keiyakudata import KeiyakuDataset, KeiyakuDataLoader
from keiyakuai import KeiyakuAI
from keiyakumodelfactory import KeiyakuModelFactory
import os
import datetime
import sys

starttime=datetime.datetime.now().strftime('%Y%m%d%H%M%S')

studydata_path = r".\data\keiyaku_study.csv"
testdata_path = r".\data\keiyaku_test.csv"
save_dir = r".\savedir"
epoch_num = 20

model_name = KeiyakuModelFactory.DEFAULT_MODEL_NAME
if len(sys.argv) >= 2:
    model_name = sys.argv[1]

keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel(model_name, loadweight=False)

study_data = KeiyakuDataset(studydata_path, False, 256, 6, tokenizer)
study_loader = KeiyakuDataLoader(study_data, True, 20)
test_data = KeiyakuDataset(testdata_path, False, 256, 6, tokenizer)
test_loader = KeiyakuDataLoader(test_data, False, 20)

save_dir = os.path.join(save_dir, starttime + "_" + model.model_name)

ai = KeiyakuAI(keiyakumodel, save_dir)
ai.train_model(study_loader, test_loader, 20)
