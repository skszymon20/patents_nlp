class CFG:
    model_name = 'bert-base-uncased'
    dropout = 0.1
    criterion = 'BCEWithLogitsLoss'
    hidden_size = 768
    scheduler = 'cosine'
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    eps = 1e-6
    betas = (0.9, 0.999)
    num_warmup_steps = 0
    num_cycles = 0.5
    num_epochs = 5
    batch_size = 16
    tokenizer_max_length = 64  # max length found in the whole dataset was 58
    nlastlinear = 1
    wandb = True
