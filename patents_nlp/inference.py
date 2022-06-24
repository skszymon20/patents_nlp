import time
import torch
import numpy as np
from patents_nlp.cfg import CFG
from patents_nlp.model import MyModel
from patents_nlp.preprocess import preprocess_test
from torch.utils.data import DataLoader


def predict(model: MyModel, dataloader: DataLoader, device: str)\
        -> np.ndarray:
    """Performs predictions on the test dataloader

    Args:
        model (MyModel): Our model from model.py
        dataloader (torch.utils.data.DataLoader): Dataloader created from test
            dataset
        device (str): one of ("gpu:0", "cpu")

    Returns:
        np.ndarray: array of predictions made by model.
    """
    since = time.time()
    sigmoid = torch.nn.Sigmoid()
    alllabels = np.array([])

    model.eval()   # Set model to evaluate mode

    allpreds = np.array([])
    alllabels = np.array([])
    # Iterate over data.
    for inputs, labels in dataloader:
        for k in inputs:
            inputs[k] = inputs[k].to(device)
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
            alllabels = np.concatenate(
                (alllabels, labels.cpu().detach().numpy()))

    time_elapsed = time.time() - since
    print('Inference complete in', end=' ')
    print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return allpreds


def inference_pipeline(model: MyModel, device: str = "cpu") -> None:
    """Performs inference based on CFG.test_location

    Args:
        model (MyModel): My Model from model.py
        device (str, optional): one of ("gpu:0", "cpu"). Defaults to "cpu".
    """
    dataloader, table = preprocess_test(CFG.test_location, return_df=True)
    predictions = predict(model, dataloader, device)
    table.drop(columns=['text'], inplace=True)
    table['score'] = predictions
    table.to_csv(CFG.submission_location, index=False)
