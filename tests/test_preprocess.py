from multiprocessing.spawn import prepare
from random import sample
import pytest
from transformers import AutoTokenizer
from patents_nlp.cfg import CFG

from patents_nlp.preprocess import Dataset, prepare_datatable, preprocess_train
import pandas as pd
import torch


def test_prepare_datatable():
    table = pd.read_csv("data/train.csv").head()
    colnames = ['id', 'text', 'score']
    prepare_datatable(table, colnames)
    for colname in colnames:
        assert colname in table.columns
        assert not any(table[colname].isna())


def test_preprocess_train():
    traindl, validdl = preprocess_train()
    assert type(traindl) == torch.utils.data.dataloader.DataLoader
    sampleX = traindl.dataset[0][0]
    assert 'input_ids' in sampleX
    assert sampleX['input_ids'].shape[0] == CFG.tokenizer_max_length
    print(sampleX)
