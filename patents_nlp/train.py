from torch.optim import AdamW
import wandb
from patents_nlp.cfg import CFG
from patents_nlp.model import MyModel
from patents_nlp.preprocess import preprocess_train
import time
import copy
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from torch.nn.modules.loss import BCEWithLogitsLoss
import numpy as np
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


def get_optimizer_parameters(model: MyModel, encoder_lr: float,
                             decoder_lr: float,
                             weight_decay: float = 0.0) -> list:
    """Returns optimizer parameters

    Args:
        model (MyModel): MyModel from model.py
        encoder_lr (float): encoder learning rate
        decoder_lr (float): decoder learning rate
        weight_decay (float, optional): Weight decay used by some modules.
            Defaults to 0.0.

    Returns:
        list: optimizer parameters
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters()
                    if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters()
                    if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def get_scheduler(cfg: CFG, optimizer: AdamW, num_train_steps: int)\
                  -> LambdaLR:
    """returns scheduler from transformers library

    Returns:
        LambdaLR: Learning rate scheduler.
    """
    if CFG.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif CFG.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler


def train_model(model: MyModel, criterion: BCEWithLogitsLoss,
                optimizer: AdamW, scheduler: LambdaLR,
                dataloaders: dict, dataset_sizes: dict, device: torch.device,
                num_epochs: int = 25) -> tuple:
    """Trains the model.

    Args:
        model (MyModel): My Model from models.py
        criterion (BCEWithLogitsLoss): loss to be used during training
        optimizer (AdamW): oprimizer to optimize model's parameters
        scheduler (LambdaLR): learning rate scheduler
        dataloaders (dict): 'train' and 'val' dataloaders from
            torch.utils.data.DataLoader
        dataset_sizes (dict): sizes of 'train' and 'val' datasets
        device (torch.device): one of ("gpu:0", "cpu")
        num_epochs (int, optional): number of training epochs. Defaults to 25.

    Returns:
        tuple: model ,bce losses, Pearsonr losses (on validation dataset)
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_personr = 0.0
    sigmoid = torch.nn.Sigmoid()
    bcelosses = {"train": [], "val": []}  # type: dict
    pearsonrlosses = {"train": [], "val": []}  # type: dict

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
                for k in inputs:
                    inputs[k] = inputs[k].to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = sigmoid(outputs)
                    labels = labels.view(-1, CFG.nlastlinear)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if CFG.wandb:
                    wandb.log({f'step_loss_{phase}': loss.item()})
                running_loss += loss.item() * labels.shape[0]
                if not allpreds.size:
                    allpreds = preds.detach().cpu().numpy()
                else:
                    allpreds = np.concatenate(
                        (allpreds, preds.cpu().detach().numpy()))
                if not alllabels.size:
                    alllabels = labels.detach().cpu().numpy()
                else:
                    alllabels = np.concatenate(
                        (alllabels, labels.cpu().detach().numpy()))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            bcelosses[phase].append(epoch_loss)
            epoch_personr = pearsonr(allpreds.flatten(), alllabels.flatten())[
                0]  # there is single output of linear -> flatten
            pearsonrlosses[phase].append(epoch_personr)

            if CFG.wandb:
                wandb.log({f"epoch_personr_{phase}": epoch_personr})
                wandb.log({f"epoch_loss_{phase}": epoch_loss})
            print(f'{phase} Loss: {epoch_loss:.4f}', end=' ')
            print(f'Pearsonr: {epoch_personr:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_personr > best_personr:
                best_personr = epoch_personr
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in', end=' ')
    print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val personr: {best_personr:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, bcelosses, pearsonrlosses


def setup_training(traindl: DataLoader,
                   validdl: DataLoader) -> tuple:
    """setups training optimizer scheduler and so.
    Should be used then for train_model function.

    Args:
        traindl (DataLoader): dataloader from training dataset
        validdl (DataLoader): dataloader from validation dataset

    Raises:
        ValueError: if criterion is not 'BCEWithLogitsLoss'.

    Returns:
        tuple: (model, criterion, optimizer, scheduler, dataloaders,
            dataset_sizes, device, num_epochs)
    """
    model = MyModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloaders = {"val": validdl, "train": traindl}
    dataset_sizes = {"val": len(validdl.dataset),  # type: ignore
                     "train": len(traindl.dataset)}  # type: ignore
    optimizer_parameters = get_optimizer_parameters(
        model, CFG.encoder_lr, CFG.decoder_lr)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr,
                      eps=CFG.eps, betas=CFG.betas)
    num_train_steps = int(len(traindl) / CFG.batch_size * CFG.num_epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    if CFG.criterion == "BCEWithLogitsLoss":
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        raise ValueError("criterion should be BCEWithLogitsLoss")
    if CFG.wandb:
        wandb.watch(model, log_freq=100)
    return (model, criterion, optimizer, scheduler, dataloaders,
            dataset_sizes, device, CFG.num_epochs)


if __name__ == "__main__":
    traindl, validdl = preprocess_train(command="todataloaders")
    setup_training(traindl, validdl)
