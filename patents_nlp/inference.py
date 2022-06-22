import time
import torch
import numpy as np
from patents_nlp.cfg import CFG
from patents_nlp.preprocess import preprocess_test
import pandas as pd


def predict(model, dataloader, device):
    since = time.time()
    sigmoid = torch.nn.Sigmoid()

    model.eval()   # Set model to evaluate mode

    allpreds = np.array([])
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = sigmoid(outputs)

        if not allpreds.size:
            allpreds = preds.detach().cpu().numpy()
        else:
            allpreds = np.concatenate((allpreds, preds.cpu().detach().numpy()))
        if not alllabels.size:
            alllabels = labels.detach().cpu().numpy()
        else:
            alllabels = np.concatenate((alllabels, labels.cpu().detach().numpy()))

    time_elapsed = time.time() - since
    print(f'Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return allpreds


def inference_pipeline(model, device="cpu"):
    dataloader, table = preprocess_test(CFG.test_location, return_df=True)
    predictions = predict(model, dataloader, device)
    table.drop(columns=['text'], inplace=True)
    table['score'] = predictions
    table.to_csv(CFG.submission_location, index=False)
