from torch.nn import BCEWithLogitsLoss


class CFG:
    model_name = 'bert-base-uncased'
    droupout = 0.1
    criterion = BCEWithLogitsLoss()
