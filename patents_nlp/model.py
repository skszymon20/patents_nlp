import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torch
from patents_nlp.cfg import CFG
from patents_nlp.preprocess import preprocess_train


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(CFG.model_name)
        self.attention = nn.Sequential(
            nn.Linear(CFG.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(CFG.dropout)
        self.linear = nn.Linear(CFG.hidden_size, CFG.nlastlinear)

    def forward(self, x: dict) -> torch.tensor:
        x = self.model(**x)
        last_hidden_states = x[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        x = self.dropout(feature)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = MyModel()
    traindl, validdl = preprocess_train(command='todataloaders')
    res = model(traindl.dataset[0])
    print(res)
