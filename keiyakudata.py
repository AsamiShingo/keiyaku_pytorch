import pandas as pd
import threading
import os
import subprocess
import numpy as np
from transformersbase import TransformersTokenizerBase
import torch
from torch.utils.data import Dataset, DataLoader

class KeiyakuDataset(Dataset):
    _CSV_HEADER = ["ファイル", "行数", "カテゴリ", "文章グループ", "分類", "条文分類", "文章"]

    create_keiyaku_data_mutex = threading.Lock()

    def __init__(self, file_path, is_predict, seq_len, category_num, tokenizer: TransformersTokenizerBase):
        self.file_path = file_path
        self.is_predict = is_predict
        self.seq_len = seq_len
        self.category_num = category_num
        self.tokenizer = tokenizer
        
        self.df = pd.read_csv(self.file_path, sep=',')
        
        header = list(self.df.columns.values)
        if header != self._CSV_HEADER:
            raise ValueError("csvfile header error(path={})".format(self.file_path))
        
        self.inputs, self.outputs = self.__load_datas(self.df, self.is_predict, self.seq_len, self.tokenizer)
        
    def get_header(self):
        return self._CSV_HEADER

    def get_datas(self):
        return self.df.values
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input = self.inputs[index]
        ouput = self.outputs[index]
        
        intensors = []
        outtensors = []
        
        input = self.tokenizer.keiyaku_encode(input, self.seq_len)
        intensors = [ torch.tensor(data) for data in input ]
        
        outtensors.append(torch.tensor([float(ouput[0])]))
        outtensors.append(torch.from_numpy(np.eye(self.category_num)[int(ouput[1])].astype(np.float32)).clone())
        
        return intensors, outtensors
    
    def __load_datas(self, df, is_predict, seq_len, tokenizer: TransformersTokenizerBase):    
        inputs = []
        outputs = []

        bef_data = None        
        for index, data in df.iterrows():
            input = []
            output = [-1, -1]
            
            if pd.isnull(data["カテゴリ"]) or pd.isnull(data["文章グループ"]):
                output[0] = -1
            elif bef_data is None:
                output[0] = 1 
            elif data["ファイル"] == bef_data["ファイル"] and data["カテゴリ"] == bef_data["カテゴリ"] and data["文章グループ"] == bef_data["文章グループ"]:
                output[0] = 0
            elif data["ファイル"] != bef_data["ファイル"]:
                output[0] = 1
                bef_data = None
            else:
                output[0] = 1
                
            if pd.isnull(data["分類"]):
                output[1] = -1
            else:
                output[1] = int(data["分類"])
                
            if bef_data is not None:
                input1 = data["文章"]
                input2 = bef_data["文章"] if not pd.isnull(bef_data["文章"]) else ""
            else:
                input1 = data["文章"] if not pd.isnull(data["文章"]) else ""
                input2 = ""
            
                
            input = tokenizer.get_keiyaku_indexes(input1 if not pd.isnull(input1) else "", input2 if not pd.isnull(input2) else "", seq_len)
            
            if is_predict == True or (output[0] != -1 and output[1] != -1):
                inputs.append(input)
                outputs.append(output)
            
            bef_data = data
            
        return inputs, outputs
    
    @classmethod
    def create_keiyaku_data(cls, srcfilepath, desttxtpath, destcsvpath):
        cls.create_keiyaku_data_mutex.acquire()

        tool_path=os.path.join(os.path.dirname(__file__), "tool", "xdoc2txt.exe")
        with open(desttxtpath, "w") as f:
            proc = subprocess.Popen([tool_path, srcfilepath], shell=True, stdout=f, stderr=subprocess.DEVNULL)
            proc.communicate()

        cls.create_keiyaku_data_mutex.release()

        datas = []
        with open(desttxtpath) as f:
            for col, line in enumerate(f):
                line=line.rstrip('\n')
                datas.append([desttxtpath, col+1, "", "", "", "", line])
        
        df = pd.DataFrame(datas, columns=cls._CSV_HEADER)
        df.to_csv(destcsvpath, encoding="UTF-8", index=False)
    
class KeiyakuDataLoader:
    def __init__(self, dataset:KeiyakuDataset, is_train, batch_size):
        self.dataset = dataset
        self.is_train = is_train
        self.batch_size = batch_size
        
    def __call__(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.is_train)
    
