from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torch
from cfg import CFG


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(CFG.model_name)
        self.dropout = nn.Dropout(CFG.dropout)
        self.linear = nn.Linear(768, 1)

    def forward(self, x: dict) -> torch.tensor:
        input_ids, attention_mask = x['input_ids'], x['attention_mask']
        _, x = self.model(input_ids, attention_mask)
        x = self.dropout(x)
        x = self.linear(x)
        return x
