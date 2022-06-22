import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import json
from evaluationscore import EvaluationScore, EvaluationScoreOutput
from keiyakumodel import KeiyakuModel
from keiyakudata import KeiyakuDataLoader

class KeiyakuLearnScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, learn_rate_percent=0.5, learn_rate_epoch=2):
        self.learn_rate_percent = learn_rate_percent
        self.learn_rate_epoch = learn_rate_epoch
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):        
        result = [ base_lr * (self.learn_rate_percent ** (self.last_epoch // self.learn_rate_epoch)) for base_lr in self.base_lrs ]
        return result
    
class KeiyakuAI:
    DEFAULT_WEIGHT_FILE_NAME = "weight.dat"
    MODEL_SUMMARY_FILE_NAME = "model_summary.txt"
    PARAM_FILE_NAME = "model_param.txt"
    
    def __init__(self, model:KeiyakuModel, save_dir):
        self.model = model
        self.save_dir = save_dir
        
        #学習設定
        self.batch_size = 20
        self.pre_epoch_num = 5
        self.pre_learn_rate= 0.001
        self.learn_rate_init= 0.0001
        self.learn_rate_epoch = 2
        self.learn_rate_percent = 0.5
        
        self.pre_optimizer = optim.Adam(self.model.parameters(), lr=self.pre_learn_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate_init)
        self.criteron = [nn.BCELoss(), nn.CrossEntropyLoss()]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_weight(self, weight_file=DEFAULT_WEIGHT_FILE_NAME):
        load_path = os.path.join(self.save_dir, weight_file)
        self.model.load_weight(load_path)
        
    def train_model(self, train_datas:KeiyakuDataLoader, test_datas:KeiyakuDataLoader, epoch_num):        
        #モデル情報保存
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, self.MODEL_SUMMARY_FILE_NAME), "w") as fp:
            fp.write(str(self.model))

        #モデル学習(全結合層)
        self.model.set_full_train(False)
        self._train_model(self.pre_optimizer, train_datas, None, self.pre_epoch_num, None)

        #モデル学習(全体)
        self.model.set_full_train(True)
        self._train_model(self.optimizer, train_datas, test_datas, epoch_num, self.save_dir)

        #モデル結果保存
        self.model.save_weight(os.path.join(self.save_dir, self.DEFAULT_WEIGHT_FILE_NAME))

    def _train_model(self, optimizer, train_datas:KeiyakuDataLoader, test_datas:KeiyakuDataLoader, epoch_num, save_dir):
        self.model.to(self.device)
        learn_scheduler = KeiyakuLearnScheduler(optimizer, -1, self.learn_rate_percent, self.learn_rate_epoch) 
        
        train_moods = [ True, False ]
        if test_datas is None:
            train_moods = [ True ]
            
        for epoch in range(epoch_num+1):
            for is_train in train_moods:
                score1 = EvaluationScore(class_num=1)
                score2 = EvaluationScore(class_num=self.model.output_class1_num)
                
                if is_train == True:
                    if epoch == 0:
                        continue
                    
                    dataloader = train_datas
                    self.model.train()
                else:
                    dataloader = test_datas
                    self.model.eval()
                
                for inputs, labels in tqdm(dataloader()):
                    inputs = [ input.to(self.device) for input in inputs ]
                    labels = [ label.to(self.device) for label in labels ]
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(is_train):
                        outputs = self.model(inputs)
                        
                        loss1 = self.criteron[0](outputs[0], labels[0])
                        loss2 = self.criteron[1](outputs[1], labels[1])
                        
                        score1.update_state(labels[0], outputs[0], loss1)
                        score2.update_state(labels[1], outputs[1], loss2)
                        
                        if is_train == True:
                            loss = loss1 + loss2
                            loss.backward()
                            optimizer.step()
                            
                if is_train == True:
                    learn_scheduler.step()
                    self.model.save_weight(os.path.join(self.save_dir, "wieght_{}_{:.3f}.dat".format(epoch, loss)))
                    
                self._output_score(is_train, score1, score2)
    
    def _output_score(self, is_train, score1:EvaluationScore, score2:EvaluationScore):
        print("is_train:{}, loss1:{:.3f}, precision1:{:.3f}, recall1:{:.3f}, fvalue1:{:.3f}, loss2:{:.3f}, precision2:{:.3f}, recall2:{:.3f}, fvalue2:{:.3f}".format(
            "学習" if is_train == True else "検証", 
            score1.get_loss(), score1.get_precision(), score1.get_recall(), score1.get_fvalue(),
            score2.get_loss(), score2.get_precision(), score2.get_recall(), score2.get_fvalue()))
        
    def predict(self, datas:KeiyakuDataLoader):
        self.model.to(self.device)        
        self.model.eval()
        
        result = None
        for inputs, _ in tqdm(datas()):
            inputs = [ input.to(self.device) for input in inputs ]
            outputs = self.model(inputs)
            outputs = (outputs[0].detach().cpu().numpy().copy(), outputs[1].detach().cpu().numpy().copy())
            if result is None:
                result = outputs
            else:
                result = (np.concatenate((result[0], outputs[0]), 0), np.concatenate((result[1], outputs[1]), 0))
            
        return result

    def _create_paramfile(self):
        with open(os.path.join(self.save_dir, self.PARAM_FILE_NAME), "w", encoding="utf-8") as f:
            data = {}
            data["batch_size"] = self.batch_size
            data["pre_epoch_num"] = self.pre_epoch_num
            data["pre_learn_rate"] = self.pre_learn_rate
            data["learn_rate_init"] = self.learn_rate_init
            data["learn_rate_epoch"] = self.learn_rate_epoch
            data["learn_rate_percent"] = self.learn_rate_percent
            json.dump(data, f, ensure_ascii=False, indent=4)