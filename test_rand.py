import torch
import torch.functional as F
import torch.nn as nn
import random
from model import CNN
from utils import get_loaders, entropy, accuracy, confidence, EWC

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()

train_dl, valid_dl, forget_dl = get_loaders()

def test_rand(num_epochs, valid_dl):
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    entropy_hist_valid = [0] * num_epochs
    confidence_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += \
                    loss.item() * y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
                entropy_hist_valid[epoch] += entropy(pred)
                confidence_hist_valid[epoch] += confidence(pred)

            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
            entropy_hist_valid[epoch] /= len(valid_dl.dataset)
            confidence_hist_valid[epoch] /= len(valid_dl.dataset)

            print(f'----- Epoch {epoch+1} -----\n'
            f' Valid Acc: {accuracy_hist_valid[epoch]:.4f}\n'
            f' Valid Loss: {loss_hist_valid[epoch]:.4f}\n'
            f' Valid Entropy: {entropy_hist_valid[epoch]:.4f}\n'
            f' Valid Confidence: {confidence_hist_valid[epoch]:.4f}\n'
            f'----------------------\n')

test_rand(1, valid_dl)