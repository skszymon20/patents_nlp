from transformers import AdamW, AutoTokenizer
from patents_nlp.cfg import CFG
from patents_nlp.model import MyModel
from patents_nlp.preprocess import Dataset, prepare_datatable, preprocess_train
from patents_nlp.train import get_optimizer_parameters, get_scheduler, setup_training, train_model
import torch


def test_get_optimizer_params():
    model = MyModel()
    encoderlr = 2e-5
    decoderlr = 2e-5
    optparams = get_optimizer_parameters(model, encoderlr, decoderlr)
    assert len(optparams) == 3
    assert optparams[-1]['params'][-2].shape == torch.Size([1, CFG.hidden_size])
    assert optparams[-1]['params'][-1].shape == torch.Size([1])


def test_get_scheduler():
    model = MyModel()
    optimizer_parameters = get_optimizer_parameters(
        model, CFG.encoder_lr, CFG.decoder_lr)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr,
                      eps=CFG.eps, betas=CFG.betas)
    get_scheduler(CFG, optimizer, 10)


def test_setup_training():
    traindl, validdl = preprocess_train(filename="./data/train_small.csv")
    mysetup = setup_training(traindl, validdl)
    assert len(mysetup) == 8


def test_train_model():
    traindl, validdl = preprocess_train(filename="./data/train_small.csv")
    mysetup = setup_training(traindl, validdl)
    model, criterion, optimizer, scheduler, dataloaders,\
        dataset_sizes, device, num_epochs = mysetup
    mysetup = list(mysetup)
    device = "cpu"
    model.to(device)
    mysetup[-2] = device
    num_epochs = 1
    mysetup[-1] = num_epochs
    model, bcelosses, pearsonrlosses = train_model(*mysetup)
    for k in bcelosses:
        assert len(bcelosses[k]) == 1
    for k in pearsonrlosses:
        assert len(bcelosses[k]) == 1
