import pytest
import shutil
from keiyakudata import KeiyakuData

class TestKeiyakuData:
	
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
            f.write("D21,D22,,D24,2,7,D25\n")
            f.write("D21,D22,D23,,2,3,D25\n")
            f.write("D21,D22,D23,D24,2,3,D25\n")
            f.write("D21,D22,D23,D24,2,3,D25\n")

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

    def test_init_error(self, keiyaku_file_error):
        with pytest.raises(ValueError):
            KeiyakuData(keiyaku_file_error)


    def test_get_header(self, keiyaku_file):
        keiyaku_data = KeiyakuData(keiyaku_file)

        datas = keiyaku_data.get_header()

        assert len(datas) == 9
        assert datas[0] == "ファイル"
        assert datas[1] == "行数"
        assert datas[2] == "カテゴリ"
        assert datas[3] == "文章グループ"        
        assert datas[4] == "分類"
        assert datas[5] == "条文分類"
        assert datas[6] == "文章"
        assert datas[7] == "前文章"
        assert datas[8] == "グループ判定"

    def test_get_datas(self, keiyaku_file):
        keiyaku_data = KeiyakuData(keiyaku_file)

        datas = keiyaku_data.get_datas()

        assert datas.shape == (10, 9)
        assert datas[0][0] == "D11"
        assert datas[1][1] == "D12"
        assert datas[2][2] == "D13"
        assert datas[3][3] == "D24"
        assert datas[4][4] == 4
        assert datas[5][5] == 6

        assert datas[0][6] == "D15"
        assert datas[0][7] == ""
        assert datas[1][6] == "D25"
        assert datas[1][7] == "D15"
        assert datas[9][6] == "D25"
        assert datas[9][7] == "D25"

        assert datas[0][8] == 1
        assert datas[1][8] == 0
        assert datas[2][8] == 1
        assert datas[3][8] == 1
        assert datas[4][8] == 0
        assert datas[5][8] == 1
        assert datas[6][8] == -1
        assert datas[7][8] == -1
        assert datas[8][8] == 1
        assert datas[9][8] == 0

    def test_get_group_datas(self, test_keiyakudata: KeiyakuData, mocker):
        tokenizer_mock = mocker.MagicMock()
        tokenizer_mock.get_keiyaku_indexes = mocker.Mock(return_value=[1, 2, 3, 4, 5, 6, 7, 8])
        
        datas = test_keiyakudata.get_group_datas(tokenizer_mock, 8)

        assert len(datas) == 1136
        assert len(datas[0]) == 2
        assert len(datas[0][1]) == 3
        for data in datas:
            assert data[0] == [1, 2, 3, 4, 5, 6, 7, 8]
            assert data[1][0] == -1 or data[1][0] == 0 or data[1][0] == 1
            assert -1 <= data[1][1] and data[1][1] <= 5
            assert -1 <= data[1][2] and data[1][2] <= 6

    def test_get_study_group_datas(self, test_keiyakudata: KeiyakuData, mocker):
        tokenizer_mock = mocker.MagicMock()
        tokenizer_mock.get_keiyaku_indexes = mocker.Mock(return_value=[1, 2, 3, 4, 5, 6, 7, 8])
        datas = test_keiyakudata.get_study_group_datas(tokenizer_mock, 16)

        assert len(datas) == 299
        assert len(datas[0]) == 2
        assert len(datas[0][1]) == 3
        for data in datas:
            assert data[0] == [1, 2, 3, 4, 5, 6, 7, 8]
            assert data[1][0] == -1 or data[1][0] == 0 or data[1][0] == 1
            assert -1 <= data[1][1] and data[1][1] <= 5
            assert -1 <= data[1][2] and data[1][2] <= 6