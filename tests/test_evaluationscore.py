import pytest
import numpy as np
import torch
import os
import pandas as pd
from evaluationscore import EvaluationScore, EvaluationScoreOutput

class TestEvaluationScore:
    def test_single_score(self):
        evaluation_score = EvaluationScore(class_num=1)
        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))
        assert evaluation_score.tp == 1
        assert evaluation_score.tn == 2
        assert evaluation_score.fp == 2
        assert evaluation_score.fn == 2

        evaluation_score = EvaluationScore(class_num=1, single_judge=0.3)
        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))
        assert evaluation_score.tp == 2
        assert evaluation_score.tn == 1
        assert evaluation_score.fp == 3
        assert evaluation_score.fn == 1
        
    def test_multi_score(self):
        evaluation_score = EvaluationScore(class_num=4)

        y_true1 = np.array([[  0,   0,   1,   0], [  0,   0,   0,   1], [  1,   0,   0,   0]])
        y_pred1 = np.array([[0.2, 0.1, 0.6, 0.1], [0.0, 0.6, 0.4, 0.0], [0.2, 0.4, 0.2, 0.2]])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.tp == 1
        assert evaluation_score.tn == 7
        assert evaluation_score.fp == 2
        assert evaluation_score.fn == 2

    def test_tp(self):
        evaluation_score = EvaluationScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.tp == 1

        y_true2 = np.array([  1,   1])
        y_pred2 = np.array([1.0, 1.0])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))
        
        assert evaluation_score.tp == 3

    def test_tn(self):
        evaluation_score = EvaluationScore()

        y_true1 = np.array([  0,   0,   0,   0,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.tn == 2

        y_true2 = np.array([  1,   1,   1,   0])
        y_pred2 = np.array([0.4, 0.6, 1.0, 0.0])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))
        
        assert evaluation_score.tn == 3

    def test_fp(self):
        evaluation_score = EvaluationScore()
        
        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.fp == 2

        y_true2 = np.array([  1,   0])
        y_pred2 = np.array([1.0, 1.0])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))
        
        assert evaluation_score.fp == 3

    def test_fn(self):
        evaluation_score = EvaluationScore()

        y_true1 = np.array([  0,   0,   0,   0,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.fn == 1

        y_true2 = np.array([  1,   1,   1,   1])
        y_pred2 = np.array([0.4, 0.6, 1.0, 0.0])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))
        
        assert evaluation_score.fn == 3
        
    def test_get_accuracy(self):
        evaluation_score = EvaluationScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.get_accuracy() == 0.5

        y_true2 = np.array([  1,   0])
        y_pred2 = np.array([1.0, 0.0])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))

        assert evaluation_score.get_accuracy() == 0.6

    def test_get_precision(self):
        evaluation_score = EvaluationScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.get_precision() == 0.5

        y_true2 = np.array([  1,   1,   1,   1,   1,   1])
        y_pred2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))

        assert evaluation_score.get_precision() == 0.8
    
    def test_get_recall(self):
        evaluation_score = EvaluationScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))

        assert evaluation_score.get_recall() == 0.5

        y_true2 = np.array([  1,   1,   1,   1,   1,   1])
        y_pred2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.4])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))

        assert evaluation_score.get_recall() == 0.7

    def test_get_fvalue(self):
        evaluation_score = EvaluationScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        evaluation_score.update_state(torch.from_numpy(y_true1), torch.from_numpy(y_pred1))
        
        assert evaluation_score.get_fvalue() == 0.5

        y_true2 = np.array([  0,   1,   1,   1,   1,   1,   0,   1])
        y_pred2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 0.4])
        evaluation_score.update_state(torch.from_numpy(y_true2), torch.from_numpy(y_pred2))

        assert evaluation_score.get_fvalue() == 0.6

    def test_reset_states(self):
        y_true = np.array([  0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0])
        y_pred = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        evaluation_score = EvaluationScore()        
        evaluation_score.update_state(torch.from_numpy(y_true), torch.from_numpy(y_pred))
        assert evaluation_score.get_accuracy() == 0.55
        
        evaluation_score.reset_states()
        evaluation_score.update_state(torch.from_numpy(np.array([1])), torch.from_numpy(np.array([1])))
        assert evaluation_score.get_accuracy() == 1
        
    def test_zero_divide(self):
        y_true = np.array([  0])
        y_pred = np.array([0.0])

        evaluation_score = EvaluationScore()
        evaluation_score.update_state(torch.from_numpy(y_true), torch.from_numpy(y_pred))
        assert evaluation_score.get_precision() == 0.0
        
        evaluation_score = EvaluationScore()
        evaluation_score.update_state(torch.from_numpy(y_true), torch.from_numpy(y_pred))
        assert evaluation_score.get_recall() == 0.0
        
class TestEvaluationScoreOutput:
    def test_output_csv_image(self, tmpdir):
        test_csv_path = os.path.join(tmpdir, r"test_output_csv.csv")
        test_image_path = os.path.join(tmpdir, r"test_output_image.png")
        evaluationScoreOutput = EvaluationScoreOutput()
        
        score1 = EvaluationScore()
        y_true = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        score1.update_state(torch.from_numpy(y_true), torch.from_numpy(y_pred))
        evaluationScoreOutput.output_csv(test_csv_path, "TEST1", True, 1, 0.1, score1)
        
        score2 = EvaluationScore()
        y_true = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        score2.update_state(torch.from_numpy(y_true), torch.from_numpy(y_pred))
        evaluationScoreOutput.output_csv(test_csv_path, "TEST1", True, 2, 0.2, score2)
            
        score3 = EvaluationScore()
        y_true = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        score3.update_state(torch.from_numpy(y_true), torch.from_numpy(y_pred))
        evaluationScoreOutput.output_csv(test_csv_path, "TEST2", False, 1, 0.3, score3)
        
        score4 = EvaluationScore()
        y_true = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        score4.update_state(torch.from_numpy(y_true), torch.from_numpy(y_pred))
        evaluationScoreOutput.output_csv(test_csv_path, "TEST2", False, 2, 0.4, score4)
        
        df = pd.read_csv(test_csv_path, sep=',')
        assert list(df.columns.values) == EvaluationScoreOutput.SAVE_ROWS
        
        assert df.iloc[0, 0] == "TEST1"
        assert df.iloc[0, 1] == True
        assert df.iloc[0, 2] == 1
        assert df.iloc[0, 3] == 0.1
        assert df.iloc[0, 4] == score1.tp
        assert df.iloc[0, 5] == score1.tn
        assert df.iloc[0, 6] == score1.fp
        assert df.iloc[0, 7] == score1.fn
        assert df.iloc[0, 8] == score1.get_accuracy()
        assert df.iloc[0, 9] == score1.get_precision()
        assert df.iloc[0, 10] == score1.get_recall()
        assert df.iloc[0, 11] == score1.get_fvalue()
        
        assert df.iloc[3, 0] == "TEST2"
        assert df.iloc[3, 1] == False
        assert df.iloc[3, 2] == 2
        assert df.iloc[3, 3] == 0.4
        assert df.iloc[3, 4] == score4.tp
        assert df.iloc[3, 5] == score4.tn
        assert df.iloc[3, 6] == score4.fp
        assert df.iloc[3, 7] == score4.fn
        assert df.iloc[3, 8] == score4.get_accuracy()
        assert df.iloc[3, 9] == score4.get_precision()
        assert df.iloc[3, 10] == score4.get_recall()
        assert df.iloc[3, 11] == score4.get_fvalue()
        
        evaluationScoreOutput.output_image(test_csv_path, test_image_path, "TEST1")
        assert os.path.isfile(test_image_path)
        
    def test_output_empty_image(self, tmpdir):
        test_csv_path = os.path.join(tmpdir, r"test_output_csv.csv")
        test_image_path = os.path.join(tmpdir, r"test_output_image.png")
        evaluationScoreOutput = EvaluationScoreOutput()
        
        evaluationScoreOutput.output_image(test_csv_path, test_image_path, "TEST1")
        assert os.path.isfile(test_image_path)
        
        
        
        
        
        
        