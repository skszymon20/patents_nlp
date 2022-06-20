from torch.optim import Adam, SGD, AdamW
from patents_nlp.cfg import CFG
from patents_nlp.model import MyModel
from preprocess import Dataset, preprocess
from torch.utils.data import DataLoader
import time
import copy
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import numpy as np
from scipy.stats import pearsonr


def get_optimizer_parameters(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    if CFG.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif CFG.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_personr = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            allpreds = np.array([])
            alllabels = np.array([])
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.nn.Sigmoid()(outputs)
                    loss = criterion(outputs, labels.view(-1, CFG.nlastlinear))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * labels.shape[0]
                allpreds = np.concatenate((allpreds, preds.cpu().numpy()))
                alllabels = np.concatenate((alllabels, labels.cpu().numpy()))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_personr = pearsonr(allpreds, alllabels)[0]

            print(f'{phase} Loss: {epoch_loss:.4f} Pearsonr: {epoch_personr:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_personr > best_personr:
                best_personr = epoch_personr
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val personr: {best_personr:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def setup_training(traindl, validdl):
    model = MyModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloaders = {"val": validdl, "train": traindl}
    dataset_sizes = {"val": len(validdl.dataset), "train": len(traindl.dataset)}
    optimizer_parameters = get_optimizer_parameters(model, CFG.encoder_lr, CFG.decoder_lr)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    num_train_steps = int(len(traindl) / CFG.batch_size * CFG.num_epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    return (model, CFG.criterion, optimizer, scheduler, dataloaders,
            dataset_sizes, device, CFG.num_epochs)


if __name__ == "__main__":
    traindl, validdl = preprocess(command="todataloaders")
    train(traindl, validdl)
