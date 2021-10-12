import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from kerasscore import KerasScore
from transformersbase import TransformersBase, TransformersTokenizerBase

class KeiyakuModel:
    def __init__(self, tokenizer: TransformersTokenizerBase, output_class1_num=6):
        self.bert_model = None
        self.model = None
        self.tokenizer = tokenizer
        
        self.seq_len = 0
        self.output_class1_num = output_class1_num

        #学習設定
        self.train_data_split = 0.8
        self.batch_size = 20
        self.learn_rate_init= 0.0001
        self.learn_rate_epoch = 2
        self.learn_rate_percent = 0.5
        
        self.optimizer="adam"
        self.loss=["binary_crossentropy", "categorical_crossentropy"]
        self.metrics=[
            [KerasScore(KerasScore.TYPE_TP), KerasScore(KerasScore.TYPE_TN),
            KerasScore(KerasScore.TYPE_FP), KerasScore(KerasScore.TYPE_FN), 
            KerasScore(KerasScore.TYPE_ACCURACY), KerasScore(KerasScore.TYPE_PRECISION),
            KerasScore(KerasScore.TYPE_RECALL), KerasScore(KerasScore.TYPE_FVALUE)],
            [KerasScore(KerasScore.TYPE_TP, self.output_class1_num), KerasScore(KerasScore.TYPE_TN, self.output_class1_num),
            KerasScore(KerasScore.TYPE_FP, self.output_class1_num), KerasScore(KerasScore.TYPE_FN, self.output_class1_num), 
            KerasScore(KerasScore.TYPE_ACCURACY, self.output_class1_num), KerasScore(KerasScore.TYPE_PRECISION, self.output_class1_num),
            KerasScore(KerasScore.TYPE_RECALL, self.output_class1_num), KerasScore(KerasScore.TYPE_FVALUE, self.output_class1_num)],
            ]

    def init_model(self, bert: TransformersBase):
        self.seq_len = bert.seq_len
        self.bert_model = bert

        bert_layer = self.bert_model.get_transformers_output()
        output_tensor = tf.keras.layers.Dropout(0.5)(bert_layer)
        output_tensor1 = tf.keras.layers.Dense(1, activation='sigmoid', name="output1")(output_tensor)
        output_tensor2 = tf.keras.layers.Dense(self.output_class1_num, activation='softmax', name="output2")(output_tensor)
        self.model = tf.keras.models.Model(self.bert_model.get_inputs(), [output_tensor1, output_tensor2])

    def load_weight(self, weight_path):
        self.model.load_weights(weight_path)
        
    def train_model(self, datas, epoch_num, save_dir):
        #データ準備
        train_data_num = int(len(datas) * self.train_data_split)
        train_datas = datas[:train_data_num]
        test_datas = datas[train_data_num:]

        random.shuffle(train_datas)
        
        train_steps_per_epoch = len(train_datas) // self.batch_size
        test_steps_per_epoch = len(test_datas) // self.batch_size
        
        #モデル情報保存
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'model_summary.txt'), "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\n"))

        try:
            open(os.path.join(save_dir, 'model.json'), 'w').write(self.model.to_json())
        except NotImplementedError:
            print("to_json is NotImplemented")

        #モデル学習(全結合層)
        self.bert_model.set_trainable(False)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics='accuracy')

        self.model.fit(self._generator_data(train_datas, self.batch_size), 
            validation_data=self._generator_data(test_datas, self.batch_size),
            steps_per_epoch=train_steps_per_epoch, validation_steps=test_steps_per_epoch,
            batch_size=self.batch_size, epochs=5)

        #モデル学習(全体)
        self.bert_model.set_trainable(True)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.model.fit(self._generator_data(train_datas, self.batch_size), 
            validation_data=self._generator_data(test_datas, self.batch_size),
            steps_per_epoch=train_steps_per_epoch, validation_steps=test_steps_per_epoch,
            batch_size=self.batch_size, epochs=epoch_num, callbacks=self._get_callbacks(save_dir))

        #モデル結果保存
        testscore = self.model.evaluate(self._generator_data(test_datas, self.batch_size),
            steps=test_steps_per_epoch, batch_size=self.batch_size, verbose=2)

        self.model.save_weights(os.path.join(save_dir, 'weights_last-{:.2f}'.format(testscore[0])))

    def predict(self, datas):
        steps_per_epoch = (len(datas) // self.batch_size)
        
        mod_data_num = len(datas) % self.batch_size

        result = self.model.predict(self._generator_data(datas, self.batch_size),
            steps=steps_per_epoch, batch_size=self.batch_size)

        if mod_data_num > 0:
            mod_result = self.model.predict(self._generator_data(datas[-mod_data_num:], mod_data_num), steps=1, batch_size=mod_data_num)
            result[0] = np.concatenate([result[0], mod_result[0]])
            result[1] = np.concatenate([result[1], mod_result[1]])

        return result

    def _get_learn_rate(self, epoch):
        return self.learn_rate_init * (self.learn_rate_percent ** (epoch // self.learn_rate_epoch))

    class ResultOutputCallback(tf.keras.callbacks.Callback):
        SAVE_ROWS = ["epoch", "loss",
            "output1_loss", "output1_tp", "output1_tn", "output1_fp", "output1_fn", "output1_accuracy", "output1_precision", "output1_recall", "output1_fvalue",
            "output2_loss", "output2_tp", "output2_tn", "output2_fp", "output2_fn", "output2_accuracy", "output2_precision", "output2_recall", "output2_fvalue",
            "val_loss",
            "val_output1_loss", "val_output1_tp", "val_output1_tn", "val_output1_fp", "val_output1_fn", "val_output1_accuracy", "val_output1_precision", "val_output1_recall", "val_output1_fvalue",
            "val_output2_loss", "val_output2_tp", "val_output2_tn", "val_output2_fp", "val_output2_fn", "val_output2_accuracy", "val_output2_precision", "val_output2_recall", "val_output2_fvalue",
            ]

        def __init__(self, save_dir):
            self.df = pd.DataFrame(columns=self.SAVE_ROWS)
            self.save_csv = os.path.join(save_dir, "result.csv")

        def on_epoch_end(self, epoch, logs={}):

            values = [epoch+1] + [logs[key] for key in self.SAVE_ROWS if key != "epoch" and key in logs]

            self.df = self.df.append(pd.Series(values, index = self.SAVE_ROWS), ignore_index = True)
            self.df.to_csv(self.save_csv, index = False)

    def _get_callbacks(self, save_dir):
        callbacks = []

        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(save_dir, 'weights_{epoch:03d}-{loss:.2f}-{output1_fvalue:.2f}-{output2_fvalue:.2f}-{val_loss:.2f}-{val_output1_fvalue:.2f}-{val_output2_fvalue:.2f}'),
                monitor='val_loss',
                verbose=1,
                save_weights_only=True,
                save_best_only=True,
                mode='auto'))
        
        callbacks.append(self.ResultOutputCallback(save_dir))
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(self._get_learn_rate))

        return callbacks

    def _generator_data(self, all_datas, batch_size):
        dummy_inputs = self.tokenizer.keiyaku_encode(all_datas[0][0], self.seq_len)
        input_num = len(dummy_inputs)

        while True:
            for step in range(len(all_datas) // batch_size):
                datas = all_datas[step*batch_size:(step+1)*batch_size]

                x_outs = [ np.zeros((batch_size, self.seq_len), dtype=np.int32) for _ in range(input_num)]
                y_out1 = np.zeros((batch_size,))
                y_out2 = np.zeros((batch_size, self.output_class1_num))

                for i in range(batch_size):
                    datas_inputs = self.tokenizer.keiyaku_encode(datas[i][0], self.seq_len)
                    datas_outputs = datas[i][1]

                    for j, datas_input in enumerate(datas_inputs):
                        x_outs[j][i, :] = datas_input[:]

                    y_out1[i] = datas_outputs[0]
                    y_out2[i, :] = tf.keras.utils.to_categorical(datas_outputs[1], num_classes=self.output_class1_num)
                        
                yield x_outs, [y_out1, y_out2]