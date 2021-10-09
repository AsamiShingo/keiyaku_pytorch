import pytest
import shutil
from transformersbert import TransformersTokenizer
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

    """
    エラー用契約文章データを読みこむ場合、エラーとなること
    """
    def test_init_error(self, keiyaku_file_error):
        with pytest.raises(ValueError):
            KeiyakuData(keiyaku_file_error)

    """
    コンストラクタ
    ヘッダが想定通りであること(グループ判定が追加されている)
    """
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

    """
    get_datas
    契約文章データにある列の情報は変わらないこと
    グループ判定は下記の条件であること
    ・"カテゴリ"か"文章グループ"に値が無い場合は-1
    ・1行前のデータと"ファイル"、"カテゴリ"、"文章グループ"が同じ場合は1、違えば0
    """
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

    """
    get_group_datas
    [CLS]+"前文章"+[SEP]+"文章"のID化したリスト、グループ判定値の2列の値が取得できること
    """
    def test_get_group_datas(self, test_keiyakudata: KeiyakuData, test_transformers_tokenizer: TransformersTokenizer):
        datas = test_keiyakudata.get_group_datas(test_transformers_tokenizer, 16)

        assert len(datas) == 1136
        assert len(datas[0]) == 4
        for data in datas:
            assert len(data[0]) <= 16
            assert data[1] == -1 or data[1] == 0 or data[1] == 1
            assert -1 <= data[2] and data[2] <= 5
            assert -1 <= data[3] and data[3] <= 6

    def test_get_group_datas_word(self, test_keiyakudata: KeiyakuData, mocker):
        sp_mock = mocker.MagicMock()
        sp_mock.get_cls_idx = mocker.Mock(return_value=100)
        sp_mock.get_sep_idx = mocker.Mock(return_value=101)
        sp_mock.get_indexes = mocker.Mock(return_value=list(range(10)))

        datas = test_keiyakudata.get_group_datas(sp_mock, 14)
        # assert datas[0][0] == [100, 0, 1, 8, 9, 101, 0, 1, 8, 9, 101]
        assert datas[0][0] == [0, 1, 8, 9, 101, 0, 1, 8, 9, 101]

        datas = test_keiyakudata.get_group_datas(sp_mock, 43)
        # assert datas[0][0] == [100, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101]
        assert datas[0][0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101]

    """
    get_study_group_datas
     [CLS]+"前文章"+[SEP]+"文章"のID化したリスト、グループ判定値でグループ判定値が-1以外のデータが取得できること
    """
    def test_get_study_group_datas(self, test_keiyakudata: KeiyakuData, test_transformers_tokenizer: TransformersTokenizer):
        datas = test_keiyakudata.get_study_group_datas(test_transformers_tokenizer, 16)

        assert len(datas) == 299
        assert len(datas[0]) == 4
        for data in datas:
            assert len(data[0]) <= 16
            assert data[1] == 0 or data[1] == 1
            assert 0 <= data[2] and data[2] <= 5
            assert 0 <= data[3] and data[3] <= 6