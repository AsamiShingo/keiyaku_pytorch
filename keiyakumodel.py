import torch
import torch.nn as nn
from transformersbase import TransformersBase, TransformersTokenizerBase

class KeiyakuModel(nn.Module):
    def __init__(self, bert_model: TransformersBase, tokenizer: TransformersTokenizerBase, output_class1_num=6):
        super().__init__()
        
        self.bert_layer = bert_model.get_transformers_model()
        self.tokenizer = tokenizer
        
        self.seq_len = bert_model.seq_len
        self.bert_output_dim = bert_model.output_dim
        self.output_class1_num = output_class1_num
        
        self.droplayer = nn.Dropout(0.5)
        self.out1layer = nn.Linear(self.bert_output_dim, 1)
        self.sigmoidlayer = nn.Sigmoid()
        self.out2layer = nn.Linear(self.bert_output_dim, self.output_class1_num)
        self.softmaxlayer = nn.Softmax(dim=1)
        
    def forward(self, inputs):
        out = self.bert_layer(inputs[0], inputs[1], inputs[2])
        out = self.droplayer(out["pooler_output"])
        out1 = self.out1layer(out)
        out1 = self.sigmoidlayer(out1)
        out2 = self.out2layer(out)
        out2 = self.softmaxlayer(out2)
        
        return out1, out2
            
    def set_full_train(self, is_full_train:bool):
        for param in self.bert_layer.parameters():
            param.requires_grad = is_full_train
            
    def save_weight(self, path):
        torch.save(super().state_dict(), path)
        
    def load_weight(self, path):
        try:
            weights = torch.load(path)
        except:
            weights = torch.load(path, map_location={'cuda:0': 'cpu'})
            
        super().load_state_dict(weights)
        
