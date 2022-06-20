from torch.nn import BCEWithLogitsLoss, MSELoss
from transformers import BertModel, BertTokenizer


class CFG:
    nlastlinear = 1
    model_name = 'bert-base-uncased'
    encoder = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dropout = 0.1
    criterion = BCEWithLogitsLoss(reduction='mean')
    hidden_size = 768
    scheduler = 'cosine'
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    eps = 1e-6
    betas = (0.9, 0.999)
    num_warmup_steps = 0
    num_cycles = 0.5
    num_epochs = 25
    batch_size = 16
    tokenizer_max_length = 64  # max length found in the whole dataset was 58
