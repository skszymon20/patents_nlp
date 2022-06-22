from patents_nlp.preprocess import preprocess_train
from patents_nlp.train import train_model, setup_training
import json
from patents_nlp.cfg import CFG
import wandb
import torch

model_names = ['distilbert-base-uncased', 'bert-base-uncased', 'microsoft/deberta-v3-base']
for model_name in [model_names[-1]]:
    save_model_name = model_name.replace('/', '-')
    print(f"model_name: {model_name}")
    CFG.model_name = model_name
    cfg_dict = {key: value for key, value in CFG.__dict__.items() if not key.startswith('__') and not callable(key)}
    if CFG.wandb:
        wandb.init(project="patents-nlp-bert", entity='3ai', config=cfg_dict, name=f"First test model: {model_name}")

    traindl, validdl = preprocess_train(CFG.train_location, command='todataloaders')
    mysetup = setup_training(traindl, validdl)
    model, bcelosses, pearsonrlosses = train_model(*mysetup)
    torch.save(model.state_dict(), f'{save_model_name}_weights.pt')
    print(bcelosses)
    print(pearsonrlosses)
    # bcelosses to json
    with open(f'bcelosses_{save_model_name}.json', 'w') as fp:
        json.dump(bcelosses, fp, indent=4)
    # pearsonrlosses to json
    with open(f'pearsonrlosses_{save_model_name}.json', 'w') as fp:
        json.dump(pearsonrlosses, fp, indent=4)
    if CFG.wandb:
        wandb.finish()
