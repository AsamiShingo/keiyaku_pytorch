import pandas as pd
import threading
import os
import subprocess
from transformersbase import TransformersTokenizerBase

class KeiyakuData:
    _CSV_HEADER_CHECK = ["ファイル", "行数", "カテゴリ", "文章グループ", "分類", "条文分類", "文章"]
    _CSV_HEADER = ["ファイル", "行数", "カテゴリ", "文章グループ", "分類", "条文分類", "文章", "前文章", "グループ判定"]

    create_keiyaku_data_mutex = threading.Lock()

    def __init__(self, file_path):
        self.file_path = file_path

        self.df = pd.read_csv(self.file_path, sep=',')
        
        header = list(self.df.columns.values)
        if header != self._CSV_HEADER_CHECK:
            raise ValueError("csvfile header error(path={})".format(self.file_path))
        
        self.__data_group_set(self.df)
        

    def get_header(self):
        return self._CSV_HEADER

    def get_datas(self):
        return self.df.values

    def get_group_datas(self, tokenizer: TransformersTokenizerBase, seq_len):
        group_datas = []
        for data in self.get_datas():
            input_ids = tokenizer.get_keiyaku_indexes(data[6], data[7], seq_len)
            outputs = [ data[8], data[4], data[5] ]
            group_datas.append((input_ids, outputs))

        return group_datas

    def get_study_group_datas(self, tokenizer: TransformersTokenizerBase, seq_len):
        group_datas = self.get_group_datas(tokenizer, seq_len)
        return list(filter(lambda x: x[1][0] != -1 and x[1][1] != -1, group_datas))

    def __data_group_set(self, df):
        df["前文章"] = ""
        df["グループ判定"] = 1

        bef_data = None
        for index, data in df.iterrows():
            if bef_data is None:
                pass
            elif pd.isnull(data["カテゴリ"]) or pd.isnull(data["文章グループ"]):
                df.iat[index, 7] = bef_data["文章"]
                df.iat[index, 8] = -1
            else:
                df.iat[index, 7] = bef_data["文章"]
                if data["ファイル"] == bef_data["ファイル"] and data["カテゴリ"] == bef_data["カテゴリ"] and data["文章グループ"] == bef_data["文章グループ"]:
                    df.iat[index, 8] = 0

            if pd.isnull(data["分類"]):
                df.iat[index, 4] = -1

            if pd.isnull(data["条文分類"]):
                df.iat[index, 5] = -1

            if pd.isnull(data["文章"]):
                df.iat[index, 6] = ""
                data["文章"] = ""

            bef_data = data

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
        
        df = pd.DataFrame(datas, columns=cls._CSV_HEADER_CHECK)
        df.to_csv(destcsvpath, encoding="UTF-8", index=False)