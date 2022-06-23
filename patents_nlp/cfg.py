class CFG:
    """General config class for experiments

    Attributes:
    model_name - name of model to be trained
    dropout - probability in range(0, 1) of dropout
    criterion - loss to be used
    hidden_size - hidden layer size used by encoder
    scheduler - scheduler to be used. One of ("linear", "cosine")
    encoder_lr - learning rate used by bert based encoder
    decoder_lr - learning rate used by the model parameters without encoder
    eps - for AdamW optimizer term added to the denominator to improve
        numerical stability (float)
    betas - for AdamW optimizer coefficients used for computing running\
         averages of gradient
    num_warmup_steps - the number of steps for warmup phase during training
    num_cycles - during training the number of waves in the cosine schedule
    num_epochs - number of training epochs
    batch_size - batch size used by dataloader
    tokenizer_max_length - length of the tokenized sequence.
    nlastlinear - number of model's outputs
    wandb - whether to use wandb logging
    train_location - location of the training dataset
    test_location - location of the test dataset
    submission_location - target location to put submission file

    """
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
    batch_size = 16  # was 16
    tokenizer_max_length = 64  # max length found in the whole dataset was 58
    nlastlinear = 1
    wandb = False
    train_location = './data/train.csv'
    test_location = './data/test.csv'
    submission_location = './data/mysubmission.csv'
