import torch
from torch.utils.data import DataLoader
import pandas as pd
from cfg import CFG


# TODO: Is there better way to assign subclasses
# (differentiate A11 from A12 etc-> possible underfitting)
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

dummycols = []


def prepare_datatable(table: pd.DataFrame,
                      col_names_returned=['id', 'text', 'score']):
    table['context'].replace(r'\d\d', '', regex=True, inplace=True)
    table['context_text'] = table['context'].map(codes)
    table['text'] = table['anchor'] + '[SEP]' + \
        table['target'] + '[SEP]' + table['context_text']
    return table[col_names_returned]
    # dummytargets = pd.get_dummies(table['score'])
    # for i in dummytargets:
    #     table[i] = dummytargets[i]
    # table = table[col_names_returned + list(dummytargets.columns)]
    # table.drop(columns=['score'], inplace=True)
    # global dummycols
    # dummycols = list(dummytargets.columns)
    # return table



class Dataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, tokenizer):
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
        X = self.tokenizer(
            row['train'], padding='max_length',
            max_length=CFG.tokenizer_max_length, add_special_tokens=True,
            return_offsets_mapping=False, return_tensors='pt'
        )
        for k in X:
            X[k] = X[k].squeeze()
        y = row['score']
        return (X, y)

def preprocess_test(filename='./data/test.csv', command="todataloader"):
    test_base = pd.read_csv(filename)  # len=36 rows
    table = prepare_datatable(["id", "text"])
    test_ds = Dataset(test_base, CFG.tokenizer)
    if command == 'todataset':
        return test_ds
    elif command == 'todataloader':
        test_dl = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)
        return test_dl
    else:
        raise ValueError('command must be "todataset" or "todataloader"')

def preprocess_train(filename='./data/train.csv', command="todataloaders", valid_size=0.2):
    train_base = pd.read_csv(filename)  # len=36473 (rows)
    # Cleaning Datatable to format: id, train text
    # (anchor + target + context_text), score
    table = prepare_datatable(train_base)
    # Train/Validation split
    table = table.sample(frac=1, random_state=123)
    sp0 = int(table.shape[0] * valid_size)
    train, validation = table[:sp0], table[sp0:]
    # Initialize some tokenizer
    tokenizer = CFG.tokenizer
    # Dataset initialization and test
    train_s = Dataset(train, tokenizer)
    val_s = Dataset(validation, tokenizer)
    if command == 'todatasets':
        return train_s, val_s
    elif command == 'todataloaders':
        train_dl = DataLoader(train_s, batch_size=CFG.batch_size, shuffle=True)  # check num workers
        valid_dl = DataLoader(val_s, batch_size=CFG.batch_size, shuffle=True)
        return train_dl, valid_dl
    else:
        raise ValueError('command must be "todatasets" or "todataloaders"')
