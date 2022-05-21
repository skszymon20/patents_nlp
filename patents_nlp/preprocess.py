import pandas as pd
import transformers
import torch
import numpy as np

from sklearn.model_selection import train_test_split

TRAIN_SET = './data/train.csv'
TEST_SET = './data/test.csv'

codes = {
    'A': 'Human Necessities',
    'B': 'Operations and Transport',
    'C': 'Chemistry and Metallurgy',
    'D': 'Textiles',
    'E': 'Fixed Constructions',
    'F': 'Mechanical Engineering',
    'G': 'Physics',
    'H': 'Electricity',
    'Y': 'Emerging Cross-Sectional Technologi',
}


def prepare_datatable(table: pd.DataFrame, col_names_returned=['id', 'train', 'score']):
    table['context'].replace(r'\d\d', '', regex=True, inplace=True)
    table['context_text'] = table['context'].map(codes)
    table['train'] = table['anchor'] + '[SEP]' + \
        table['target'] + '[SEP]' + table['context_text']
    return table[col_names_returned]


def init_tokenizer(model_name: str, token_max_len: int, tokenizer=transformers.AutoTokenizer,):
    return tokenizer.from_pretrained(model_name, model_max_length=token_max_len)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, tokenizer):
        'Initialization'
        self.table = table
        self.tokenizer = tokenizer

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.table)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.table.iloc[index]

        # Load data and get label
        X = self.tokenizer.tokenize(row['train'])
        y = row['score']

        return (X, y)


# Two given datasets
train_base = pd.read_csv(TRAIN_SET)  # len=36473 (rows)
test_base = pd.read_csv(TEST_SET)  # len=36 (rows)

merged = pd.concat([train_base, test_base])  # concatenating both datasets

# Cleaning Datatable to format: id, train text (anchor + target + context_text), score
table = prepare_datatable(merged)

# Train/Validation split
train, validation = train_test_split(merged, test_size=0.2)





### TEMP: Checking if everything works-> it does ###
tokenizer = init_tokenizer('bert-base-uncased', 100,
                           tokenizer=transformers.AutoTokenizer)
# print(type(tokenizer))

# tokenizer test
print(tokenizer.tokenize('abatement[SEP]forest region[SEP]Human Necessities'))

# Dataset test
train_s = Dataset(train, tokenizer)
val_s = Dataset(validation, tokenizer)
t = train_s.__getitem__(0)
v = val_s.__getitem__(0)
print(t, v)
