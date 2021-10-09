import tensorflow as tf
import tensorflow.keras.backend as K

class KerasScore(tf.keras.metrics.Metric):
    TYPE_TP = "tp"
    TYPE_TN = "tn"
    TYPE_FP = "fp"
    TYPE_FN = "fn"
    TYPE_ACCURACY = "accuracy"
    TYPE_PRECISION = "precision"
    TYPE_RECALL = "recall"
    TYPE_FVALUE = "fvalue"

    def __init__(self, name=None, class_num=1, **kwargs):
        super().__init__(name, **kwargs)
        
        self.result_name = name
        self.class_num = class_num

        self.tp = self.add_weight("tp", initializer="zeros")
        self.tn = self.add_weight("tn", initializer="zeros")
        self.fp = self.add_weight("fp", initializer="zeros")
        self.fn = self.add_weight("tn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.class_num == 1:
            tp, tn, fp, fn = self._get_single_tptnfpfn(y_true, y_pred)
        else:
            tp, tn, fp, fn = self._get_multi_tptnfpfn(y_true, y_pred, self.class_num)

        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
            
    def reset_states(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

    def result(self):

        result = 0
        if self.result_name == self.TYPE_ACCURACY:
            result = self.get_accuracy()
        elif self.result_name == self.TYPE_PRECISION:
            result = self.get_precision()
        elif self.result_name == self.TYPE_RECALL:
            result = self.get_recall()
        elif self.result_name == self.TYPE_FVALUE:
            result = self.get_fvalue()
        elif self.result_name == self.TYPE_TP:
            result = self.tp
        elif self.result_name == self.TYPE_TN:
            result = self.tn
        elif self.result_name == self.TYPE_FP:
            result = self.fp
        elif self.result_name == self.TYPE_FN:
            result = self.fn
        else:
            raise ValueError("result_name error(result_name={})".format(self.result_name))
        
        if tf.math.is_nan(result):
            result = tf.constant(0.0)

        return result

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_precision(self):
        return self.tp / (self.tp + self.fp)
    
    def get_recall(self):
        return self.tp / (self.tp + self.fn)
    
    def get_fvalue(self):
        return 2 * self.get_precision() * self.get_recall() / (self.get_precision() + self.get_recall())

    def _get_single_tptnfpfn(self, y_true, y_pred):
        tp = K.cast(K.sum(K.round(y_true * y_pred)), K.floatx())
        tn = K.sum(K.cast(K.equal(K.round(y_true + y_pred), 0), K.floatx()))
        fp = K.sum(K.cast(K.equal(K.round(y_pred) - y_true, 1), K.floatx()))
        fn = K.sum(K.cast(K.equal(y_true - K.round(y_pred), 1), K.floatx()))

        return tp, tn, fp, fn
    
    def _get_multi_tptnfpfn(self, y_true, y_pred, class_num):
        y_true = tf.math.argmax(y_true, axis=1)
        y_pred = tf.math.argmax(y_pred, axis=1)
        matrix =tf.math.confusion_matrix(y_true, y_pred, num_classes=class_num)
        
        tp = K.sum(tf.linalg.diag_part(matrix))            
        fp = K.sum(matrix) - tp            
        fn = len(y_true) - tp            
        tn = class_num * len(y_true) - (tp + fn + fp)

        tp = K.cast(tp, K.floatx())
        tn = K.cast(tn, K.floatx())
        fp = K.cast(fp, K.floatx())
        fn = K.cast(fn, K.floatx())

        return tp, tn, fp, fn

