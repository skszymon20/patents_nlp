from preprocess import preprocess_train
from train import train_model, setup_training
import json
from cfg import CFG
import wandb

model_names = ['bert-base-uncased', 'deberta-base', 'distilbert-base-uncased']
for model_name in model_names:
    CFG.model_name = model_name
    cfg_dict = {key: value for key, value in CFG.__dict__.items() if not key.startswith('__') and not callable(key)}
    if CFG.wandb:
        wandb.init(project="patents-nlp-bert", entity='3ai')
        wandb.config = cfg_dict

    traindl, validdl = preprocess_train(command='todataloaders')
    mysetup = setup_training(traindl, validdl)
    print(mysetup[4].keys())
    model, bcelosses, pearsonrlosses = train_model(*mysetup)
    print(bcelosses)
    print(pearsonrlosses)
    # bcelosses to json
    with open(f'bcelosses_{CFG.model_name}.json', 'w') as fp:
        json.dump(bcelosses, fp, indent=4)
    # pearsonrlosses to json
    with open(f'pearsonrlosses_{CFG.model_name}.json', 'w') as fp:
        json.dump(pearsonrlosses, fp, indent=4)
