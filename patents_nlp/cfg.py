from torch.nn import Sigmoid


class CFG:
    model_name = 'bert-base-uncased'
    droupout = 0.1
    criterion = Sigmoid()
