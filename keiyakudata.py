import pandas as pd
import numpy as np
from transfomersbert import TransfomersTokenizer

class KeiyakuData:
    _CSV_HEADER_CHECK = ["ファイル", "行数", "カテゴリ", "文章グループ", "分類", "条文分類", "文章"]
    _CSV_HEADER = ["ファイル", "行数", "カテゴリ", "文章グループ", "分類", "条文分類", "文章", "前文章", "グループ判定"]

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

    def get_group_datas(self, tokenizer: TransfomersTokenizer, seq_num):
        sep_idx = tokenizer.get_sep_idx()
        cut_len = int((seq_num - 3)/2)
        cut_len_herf = int(cut_len / 2)

        group_datas = []
        for data in self.get_datas():
            now_text_idx = tokenizer.get_indexes(data[6])
            bef_text_idx = tokenizer.get_indexes(data[7])

            if len(now_text_idx) > cut_len:
                now_text_idx = now_text_idx[:cut_len_herf] + now_text_idx[-cut_len_herf:]

            if len(bef_text_idx) > cut_len:
                bef_text_idx = bef_text_idx[:cut_len_herf] + bef_text_idx[-cut_len_herf:]

            group_datas.append((now_text_idx + [sep_idx] + bef_text_idx + [sep_idx], data[8], data[4], data[5]))

        return group_datas

    def get_study_group_datas(self, tokenizer: TransfomersTokenizer, seq_num):
        group_datas = self.get_group_datas(tokenizer, seq_num)
        return list(filter(lambda x: x[1] != -1 and x[2] != -1, group_datas))

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

            bef_data = data