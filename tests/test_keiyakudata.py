import pytest
import shutil
import os
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from keiyakudata import KeiyakuDataset, KeiyakuDataLoader

class TestKeiyakuDataset:
	
    """
    fixture
    契約文章データ
    """
    @pytest.fixture(scope="class")
    def keiyaku_file(self, tmpdir_factory):
        tmpdir_path = tmpdir_factory.mktemp("TestKeiyakuData")
        file_path = tmpdir_path.join("keiyaku_file.csv")
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write("ファイル,行数,カテゴリ,文章グループ,分類,条文分類,文章\n")
            f.write("D11,D12,D13,D14,0,1,D15\n")
            f.write("D11,D12,D13,D14,1,2,D25\n")
            f.write("D11,D12,D13,D24,2,3,D25\n")
            f.write("D11,D12,D23,D24,3,4,D25\n")
            f.write("D11,D22,D23,D24,4,5,D25\n")
            f.write("D21,D22,D23,D24,5,6,D25\n")
            f.write("D21,D22,,D24,2,7,D35\n")
            f.write("D21,D22,D23,,2,3,D25\n")
            f.write("D21,D22,D23,D24,,3,D25\n")
            f.write("D21,D22,,,,3,\n")

        yield file_path

        shutil.rmtree(tmpdir_path)

    """
    fixture
    エラー用契約文章データ
    """
    @pytest.fixture(scope="class")
    def keiyaku_file_error(self, tmpdir_factory):
        tmpdir_path = tmpdir_factory.mktemp("TestKeiyakuData")
        file_path = tmpdir_path.join("keiyaku_file_error.csv")
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write("ファイル,行数,カテゴリ,文章グループ,分類,条文分類,文章,ERROR\n")
            f.write("D11,D12,D13,D14,1,2,D15,ERR\n")

        yield file_path

        shutil.rmtree(tmpdir_path)

    """
    fixture
    Tokenizerモック
    """
    @pytest.fixture()
    def tokenizer_mock(self, mocker):
        tokenizer_mock = mocker.MagicMock()
        tokenizer_mock.get_keiyaku_indexes = mocker.Mock(return_value=[1, 2, 3, 4, 5, 6, 7, 8])
        return tokenizer_mock
        
    def test_init_error(self, keiyaku_file_error, tokenizer_mock):
        with pytest.raises(ValueError):
            KeiyakuDataset(keiyaku_file_error, False, 256, 6, tokenizer_mock)


    def test_get_header(self, keiyaku_file, tokenizer_mock):
        keiyaku_data = KeiyakuDataset(keiyaku_file, False, 256, 6, tokenizer_mock)

        datas = keiyaku_data.get_header()

        assert len(datas) == 7
        assert datas[0] == "ファイル"
        assert datas[1] == "行数"
        assert datas[2] == "カテゴリ"
        assert datas[3] == "文章グループ"        
        assert datas[4] == "分類"
        assert datas[5] == "条文分類"
        assert datas[6] == "文章"

    def test_get_datas(self, keiyaku_file, tokenizer_mock):
        keiyaku_data = KeiyakuDataset(keiyaku_file, False, 256, 6, tokenizer_mock)

        datas = keiyaku_data.get_datas()

        assert datas.shape == (10, 7)
        assert datas[0][0] == "D11"
        assert datas[1][1] == "D12"
        assert datas[2][2] == "D13"
        assert datas[3][3] == "D24"
        assert datas[4][4] == 4
        assert datas[5][5] == 6
        assert datas[6][6] == "D35"
    
    def test__len__(self, keiyaku_file, tokenizer_mock):
        keiyaku_data_study = KeiyakuDataset(keiyaku_file, False, 256, 6, tokenizer_mock)
        keiyaku_data_predict = KeiyakuDataset(keiyaku_file, True, 256, 6, tokenizer_mock)
        assert keiyaku_data_study.__len__() == 6
        assert keiyaku_data_predict.__len__() == 10
        
    def test__getitem__(self, keiyaku_file, mocker):
        tokenizer_mock = mocker.MagicMock()
        tokenizer_mock.keiyaku_encode = mocker.Mock(return_value=[[1, 2, 3, 4], [2, 2, 3, 4], [3, 2, 3, 4]])
        
        keiyaku_data = KeiyakuDataset(keiyaku_file, False, 256, 6, tokenizer_mock)
        inputs, outputs = keiyaku_data.__getitem__(0)
        
        assert all(inputs[0].numpy() == np.array([1, 2, 3, 4]))
        assert all(inputs[1].numpy() == np.array([2, 2, 3, 4]))
        assert all(inputs[2].numpy() == np.array([3, 2, 3, 4]))
        assert all(outputs[0].numpy() == np.array([1.0]))
        assert all(outputs[1].numpy() == np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        
        inputs, outputs = keiyaku_data.__getitem__(1)
        assert all(outputs[0].numpy() == np.array([0.0]))
        assert all(outputs[1].numpy() == np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
         
        inputs, outputs = keiyaku_data.__getitem__(2)
        assert all(outputs[0].numpy() == np.array([1.0]))
        assert all(outputs[1].numpy() == np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
         
        inputs, outputs = keiyaku_data.__getitem__(3)
        assert all(outputs[0].numpy() == np.array([1.0]))
        assert all(outputs[1].numpy() == np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
         
        inputs, outputs = keiyaku_data.__getitem__(4)
        assert all(outputs[0].numpy() == np.array([0.0]))
        assert all(outputs[1].numpy() == np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
         
        inputs, outputs = keiyaku_data.__getitem__(5)
        assert all(outputs[0].numpy() == np.array([1.0]))
        assert all(outputs[1].numpy() == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    def test_create_keiyaku_data(self, tmpdir):
        testdatadir=os.path.join(os.path.dirname(__file__), "data")
        srcfilepath=os.path.join(testdatadir, "test_keiyakudata.pdf")
        desttxtpath=os.path.join(tmpdir, "testdata.txt")
        destcsvpath=os.path.join(tmpdir, "testdata.csv")
        KeiyakuDataset.create_keiyaku_data(srcfilepath, desttxtpath, destcsvpath)

        assert os.path.exists(desttxtpath) == True
        assert os.path.exists(destcsvpath) == True

        df = pd.read_csv(destcsvpath, sep=',')
        header = list(df.columns.values)
        assert header == KeiyakuDataset._CSV_HEADER
        assert len(df.values) >= 1
        
class TestKeiyakuDataLoader:
    """
    fixture
    契約文章データ
    """
    @pytest.fixture(scope="class")
    def keiyaku_file(self, tmpdir_factory):
        tmpdir_path = tmpdir_factory.mktemp("TestKeiyakuData")
        file_path = tmpdir_path.join("keiyaku_file.csv")
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write("ファイル,行数,カテゴリ,文章グループ,分類,条文分類,文章\n")
            f.write("D11,D12,D13,D14,0,1,D15\n")
            f.write("D11,D12,D13,D14,1,2,D25\n")
            f.write("D11,D12,D13,D24,2,3,D25\n")
            f.write("D11,D12,D23,D24,3,4,D25\n")
            f.write("D11,D22,D23,D24,4,5,D25\n")
            f.write("D21,D22,D23,D24,5,6,D25\n")
            f.write("D21,D22,,D24,2,7,D35\n")
            f.write("D21,D22,D23,,2,3,D25\n")
            f.write("D21,D22,D23,D24,,3,D25\n")
            f.write("D21,D22,,,,3,\n")

        yield file_path

        shutil.rmtree(tmpdir_path)
    
    """
    fixture
    Tokenizerモック
    """
    @pytest.fixture()
    def tokenizer_mock(self, mocker):
        tokenizer_mock = mocker.MagicMock()
        tokenizer_mock.get_keiyaku_indexes = mocker.Mock(return_value=[1, 2, 3, 4, 5, 6, 7, 8])
        return tokenizer_mock
        
    def test__call__(self, keiyaku_file, tokenizer_mock):
        keiyaku_data_study = KeiyakuDataset(keiyaku_file, False, 256, 6, tokenizer_mock)
        data_loader_study = KeiyakuDataLoader(keiyaku_data_study, True, 20)
        loader = data_loader_study()
        assert isinstance(loader, DataLoader)        
        
        keiyaku_data_predict = KeiyakuDataset(keiyaku_file, True, 256, 6, tokenizer_mock)
        data_loader_predict = KeiyakuDataLoader(keiyaku_data_predict, True, 20)
        loader = data_loader_predict()
        assert isinstance(loader, DataLoader)
        
