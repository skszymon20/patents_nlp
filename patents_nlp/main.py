from preprocess import preprocess_train
from train import train_model, setup_training
import json
import torch
traindl, validdl = preprocess_train(command='todataloaders')
mysetup = setup_training(traindl, validdl)
model, bcelosses, pearsonrlosses = train_model(*mysetup)
print(bcelosses)
print(pearsonrlosses)
# bcelosses to json
with open('bcelosses.json', 'w') as fp:
    json.dump(bcelosses, fp)
# pearsonrlosses to json
with open('pearsonrlosses.json', 'w') as fp:
    json.dump(pearsonrlosses, fp)
