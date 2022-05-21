import pandas as pd
import transformers
# import torch

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


def preprocess(table, col_names_returned=['id', 'train', 'score']):
    table['context'].replace(r'\d\d', '', regex=True, inplace=True)
    table['context_text'] = table['context'].map(codes)
    table['train'] = table['anchor'] + '[SEP]' + \
        table['target'] + '[SEP]' + table['context_text']
    return table[col_names_returned]


def init_tokenizer(model_name, token_max_len, tokenizer=transformers.AutoTokenizer,):
    return tokenizer.from_pretrained(model_name, model_max_length=token_max_len)


def tokenize(tokenizer, row):
    return tokenizer.tokenize(row)


train = pd.read_csv(TRAIN_SET)

table = preprocess(train)
print(table.head())

# tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

tokenizer = init_tokenizer('bert-base-uncased', 100,
                           tokenizer=transformers.AutoTokenizer)
print(tokenize(tokenizer, 'abatement[SEP]forest region[SEP]Human Necessities'))
