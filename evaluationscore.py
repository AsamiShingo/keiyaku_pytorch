import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

class EvaluationScore:
    def __init__(self, class_num=1, single_judge=0.5):
        self.class_num = class_num
        self.single_judge = single_judge
        
        self.reset_states()
        
    def update_state(self, y_true, y_pred):
        if type(y_true) is torch.Tensor:
            y_true = y_true.detach().cpu().numpy().copy()
        
        if type(y_pred) is torch.Tensor:
            y_pred = y_pred.detach().cpu().numpy().copy()
            
        if self.class_num == 1:
            tp, tn, fp, fn = self._get_single_tptnfpfn(y_true, y_pred)
        else:
            tp, tn, fp, fn = self._get_multi_tptnfpfn(y_true, y_pred, self.class_num)

        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
            
    def reset_states(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def get_accuracy(self):        
        return round(float((self.tp + self.tn)) / (self.tp + self.tn + self.fp + self.fn), 3)

    def get_precision(self):
        return round(float(self.tp) / (self.tp + self.fp) if self.tp + self.fp > 0 else 0.0, 3)
    
    def get_recall(self):
        return round(float(self.tp) / (self.tp + self.fn) if self.tp + self.fn > 0 else 0.0, 3)
    
    def get_fvalue(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return round(2 * float(self.get_precision() * self.get_recall()) / (precision + recall) if precision + recall > 0 else 0.0, 3)

    def _get_single_tptnfpfn(self, y_true:np.ndarray, y_pred:np.ndarray):
        y_true = np.where(y_true > self.single_judge, 1, 0)
        y_pred = np.where(y_pred > self.single_judge, 1, 0)        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        tn, fp, fn, tp = cm.flatten()
        return tp, tn, fp, fn
    
    def _get_multi_tptnfpfn(self, y_true, y_pred, class_num):
        y_true = np.argmax(y_true, 1)
        y_pred = np.argmax(y_pred, 1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(class_num)))

        tp = np.sum(np.diagonal(cm))           
        fp = np.sum(cm) - tp            
        fn = len(y_true) - tp            
        tn = class_num * len(y_true) - (tp + fn + fp)

        return tp, tn, fp, fn
    
class EvaluationScoreOutput:
    SAVE_ROWS = [
        "type", "train", "epoch", "loss",
        "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "fvalue"
        ]

    def __init__(self):
        pass
    
    def output_csv(self, csv_path, type, train, epoch, loss, score:EvaluationScore):
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path, sep=',')
        else:
            df = pd.DataFrame(columns=self.SAVE_ROWS)
            
        values = [type, train, epoch, loss]
        values += [ score.tp, score.tn, score.fp, score.fn, score.get_accuracy(), score.get_precision(), score.get_recall(), score.get_fvalue() ]
        
        df = df.append(pd.Series(values, index = self.SAVE_ROWS), ignore_index = True)
        df.to_csv(csv_path, index = False)
        
    def output_image(self, csv_path, image_path, type):
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path, sep=',')
        else:
            df = pd.DataFrame(columns=self.SAVE_ROWS)
        
        df = df[df['type'] == type]
        
        val_df = df[df['train'] == False]
        train_df = df[df['train'] == True]
        
        self._create_graph(type, val_df, train_df, "epoch", ["accuracy", "precision", "recall", "fvalue"], ["loss"], image_path)
    
    def _create_graph(self, title, val_datas, train_datas, xlabel, ylabels1, ylabels2, savefile):
        # GPU環境でnp.dotがabortするため機能削除し、空ファイル作成に変更(原因不明)
        x_val_value = val_datas[xlabel].values
        x_train_value = train_datas[xlabel].values
        
        y_val_values1 = [ val_datas[label].values for label in ylabels1 ]
        y_val_values2 = [ val_datas[label].values for label in ylabels2 ]
        y_train_values1 = [ train_datas[label].values for label in ylabels1 ]
        y_train_values2 = [ train_datas[label].values for label in ylabels2 ]
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()

        for yvalue, ylabel in zip(y_val_values1, ylabels1):
            ax1.plot(x_val_value, yvalue, marker='*', label=ylabel)

        for yvalue, ylabel in zip(y_val_values2, ylabels2):
            ax2.plot(x_val_value, yvalue, marker='o', label=ylabel)
        
        for yvalue, ylabel in zip(y_train_values1, ylabels1):
            ax1.plot(x_train_value, yvalue, marker='+', label=ylabel+"(train)")

        for yvalue, ylabel in zip(y_train_values2, ylabels2):
            ax2.plot(x_train_value, yvalue, marker='.', label=ylabel+"(train)")

        ax1.set_ylim(0, 1)
        ax1.set_yticks(np.arange(0, 1.01, step=0.1))
        handler1, label1 = ax1.get_legend_handles_labels()

        ax2.set_ylim(0, 1.5)
        ax2.set_yticks(np.arange(0, 1.51, step=0.1))
        handler2, label2 = ax2.get_legend_handles_labels()
        
        ax1.set_title(title, fontname="Meiryo")
        ax1.set_xticks(x_val_value)
        ax1.legend(handler1 + handler2, label1 + label2, loc='upper left', borderaxespad=0.)
        
        plt.savefig(savefile)
        # import pathlib
        # pathlib.Path(savefile).touch()
