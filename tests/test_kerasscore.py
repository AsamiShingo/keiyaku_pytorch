import pytest
import numpy as np
from kerasscore import KerasScore
import tensorflow as tf

class TestKerasScore:
    def test_tensor_score(self):
        keras_score = KerasScore(class_num=4)

        y_true1 = np.array([[  0,   0,   1,   0], [  0,   0,   0,   1], [  1,   0,   0,   0]])
        y_pred1 = np.array([[0.2, 0.1, 0.6, 0.1], [0.0, 0.6, 0.4, 0.0], [0.2, 0.4, 0.2, 0.2]])
        y_true1 = tf.constant(y_true1)
        y_pred1 = tf.constant(y_pred1)
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.tp == 1
        assert keras_score.tn == 7
        assert keras_score.fp == 2
        assert keras_score.fn == 2

    def test_sum_score(self):
        keras_score = KerasScore(class_num=4)

        y_true1 = np.array([[  0,   0,   1,   0], [  0,   0,   0,   1], [  1,   0,   0,   0]])
        y_pred1 = np.array([[0.2, 0.1, 0.6, 0.1], [0.0, 0.6, 0.4, 0.0], [0.2, 0.4, 0.2, 0.2]])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.tp == 1
        assert keras_score.tn == 7
        assert keras_score.fp == 2
        assert keras_score.fn == 2

    def test_tp(self):
        keras_score = KerasScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.tp == 1

        y_true2 = np.array([  1,   1])
        y_pred2 = np.array([1.0, 1.0])
        keras_score.update_state(y_true2, y_pred2)
        
        assert keras_score.tp == 3

    def test_tn(self):
        keras_score = KerasScore()

        y_true1 = np.array([  0,   0,   0,   0,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.tn == 2

        y_true2 = np.array([  1,   1,   1,   0])
        y_pred2 = np.array([0.4, 0.6, 1.0, 0.0])
        keras_score.update_state(y_true2, y_pred2)
        
        assert keras_score.tn == 3

    def test_fp(self):
        keras_score = KerasScore()
        
        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.fp == 2

        y_true2 = np.array([  1,   0])
        y_pred2 = np.array([1.0, 1.0])
        keras_score.update_state(y_true2, y_pred2)
        
        assert keras_score.fp == 3

    def test_fn(self):
        keras_score = KerasScore()

        y_true1 = np.array([  0,   0,   0,   0,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.fn == 1

        y_true2 = np.array([  1,   1,   1,   1])
        y_pred2 = np.array([0.4, 0.6, 1.0, 0.0])
        keras_score.update_state(y_true2, y_pred2)
        
        assert keras_score.fn == 3
        
    def test_get_accuracy(self):
        keras_score = KerasScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.get_accuracy() == 0.5

        y_true2 = np.array([  1,   0])
        y_pred2 = np.array([1.0, 0.0])
        keras_score.update_state(y_true2, y_pred2)

        assert keras_score.get_accuracy() == 0.6

    def test_get_precision(self):
        keras_score = KerasScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.get_precision() == 0.5

        y_true2 = np.array([  1,   1,   1,   1,   1,   1])
        y_pred2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        keras_score.update_state(y_true2, y_pred2)

        assert keras_score.get_precision() == 0.8
    
    def test_get_recall(self):
        keras_score = KerasScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        keras_score.update_state(y_true1, y_pred1)

        assert keras_score.get_recall() == 0.5

        y_true2 = np.array([  1,   1,   1,   1,   1,   1])
        y_pred2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.4])
        keras_score.update_state(y_true2, y_pred2)

        assert keras_score.get_recall() == 0.7

    def test_get_fvalue(self):
        keras_score = KerasScore()

        y_true1 = np.array([  0,   0,   0,   0,   1,   1,   1,   1])
        y_pred1 = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0])
        keras_score.update_state(y_true1, y_pred1)
        
        assert keras_score.get_fvalue() == 0.5

        y_true2 = np.array([  0,   1,   1,   1,   1,   1,   0,   1])
        y_pred2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 0.4])
        keras_score.update_state(y_true2, y_pred2)

        assert keras_score.get_fvalue() == 0.6

    def test_result(self):
        y_true = np.array([  0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0])
        y_pred = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        keras_score = KerasScore(KerasScore.TYPE_ACCURACY)        
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 0.55
            
        keras_score = KerasScore(KerasScore.TYPE_PRECISION)        
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 0.6
        
        keras_score = KerasScore(KerasScore.TYPE_RECALL)
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 0.3
                
        keras_score = KerasScore(KerasScore.TYPE_FVALUE)
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 0.4
        
        keras_score = KerasScore(KerasScore.TYPE_TP)
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 3
    
        keras_score = KerasScore(KerasScore.TYPE_TN)
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 8
            
        keras_score = KerasScore(KerasScore.TYPE_FP)
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 2
            
        keras_score = KerasScore(KerasScore.TYPE_FN)
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 7

        keras_score = KerasScore("error")
        keras_score.update_state(y_true, y_pred)
        with pytest.raises(ValueError):
             keras_score.result()

    def test_reset_states(self):
        y_true = np.array([  0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0])
        y_pred = np.array([0.0, 0.4, 0.6, 1.0, 0.0, 0.4, 0.6, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        keras_score = KerasScore(KerasScore.TYPE_ACCURACY)        
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 0.55
        
        keras_score.reset_states()
        keras_score.update_state(np.array([1]), np.array([1]))
        assert keras_score.result() == 1

    def test_zero_divide(self):
        y_true = np.array([  0])
        y_pred = np.array([0.0])

        keras_score = KerasScore(KerasScore.TYPE_PRECISION)        
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 0.0
        
        keras_score = KerasScore(KerasScore.TYPE_RECALL)
        keras_score.update_state(y_true, y_pred)
        assert keras_score.result() == 0.0