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
                      col_names_returned=['id', 'train', 'score']):
    table['context'].replace(r'\d\d', '', regex=True, inplace=True)
    table['context_text'] = table['context'].map(codes)
    table['train'] = table['anchor'] + '[SEP]' + \
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


def preprocess(trainset='./data/train.csv', testset='./data/test.csv', command="todatasets", test_size=0.2):
    train_base = pd.read_csv(trainset)  # len=36473 (rows)
    test_base = pd.read_csv(testset)  # len=36 (rows)
    merged = pd.concat([train_base, test_base])  # concatenating both datasets
    # Cleaning Datatable to format: id, train text
    # (anchor + target + context_text), score
    table = prepare_datatable(merged)
    # Train/Validation split
    table = table.sample(frac=1, random_state=123)
    sp0 = int(table.shape[0] * test_size)
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



"""
# Road 2 datasets
train_base = pd.read_csv(TRAIN_SET)  # len=36473 (rows)
test_base = pd.read_csv(TEST_SET)  # len=36 (rows)

merged = pd.concat([train_base, test_base])  # concatenating both datasets

# Cleaning Datatable to format: id, train text
# (anchor + target + context_text), score
table = prepare_datatable(merged)

# Train/Validation split
train, validation = train_test_split(merged, test_size=0.2)

# Initialize some tokenizer
tokenizer = init_tokenizer('bert-base-uncased', 100,
                           tokenizer=transformers.AutoTokenizer)

# Tokenizer test
print(tokenizer.tokenize('abatement[SEP]forest region[SEP]Human Necessities'))

# Dataset initialization and test
train_s = Dataset(train, tokenizer)
val_s = Dataset(validation, tokenizer)

t = train_s.__getitem__(0)
v = val_s.__getitem__(0)
print(t, v)
"""
