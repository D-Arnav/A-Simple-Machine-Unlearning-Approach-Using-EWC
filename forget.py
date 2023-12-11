import torch
import torch.functional as F
import torch.nn as nn
import random
from model import CNN
from utils import get_loaders, entropy, accuracy, confidence, EWC
import wandb

run = wandb.init(
project="machine-unlearning",
config={
    "learning_rate": 1.5e-4,
    "epochs": 200,
})


torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()

train_dl, valid_dl, forget_dl = get_loaders()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load('weights/model.pt'))
model.to(device)

def forget(model, num_epochs, forget_dl, sample_size=200, importance=2500):
    accuracy_train = 0
    accuracy_valid = 0
    with torch.no_grad():
        for x_batch, y_batch in valid_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
            ).float()
            accuracy_valid += is_correct.sum()
        accuracy_valid /= len(valid_dl.dataset)

        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
            ).float()
            accuracy_train += is_correct.sum()
        accuracy_train /= len(train_dl.dataset)
        print(f'Initial Train Acc: {accuracy_train:.4f} Valid Acc: {accuracy_valid:.4f}')

    loss_hist_forget = [0] * num_epochs
    accuracy_hist_forget = [0] * num_epochs
    entropy_hist_forget = [0] * num_epochs
    confidence_hist_forget = [0] * num_epochs

    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    entropy_hist_train = [0] * num_epochs
    confidence_hist_train = [0] * num_epochs
    
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    entropy_hist_valid = [0] * num_epochs
    confidence_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in forget_dl:  
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch).to(device)
            
            # Uniform Prediction
            uniform_pred = 0.1 * torch.ones(*y_batch.shape, 10).to(device)
            
            # Random Prediction
            random_pred = torch.zeros(*y_batch.shape, 10)    
            random_indices = torch.tensor([random.randint(0, 9) for _ in range(y_batch.shape[0])])            
            random_pred.scatter_(1, random_indices.view(-1, 1), 1)
            random_pred = random_pred.to(device)

            size = 0
            imgs = []
            for x_b, y_b in train_dl:
                x_b, y_b = x_b.to(device), y_b.to(device)
                imgs.append(x_b)
                size += 1
                if size >= 200:
                    break
            ewc = EWC(model, imgs)
            loss = loss_fn(pred, random_pred) + importance * ewc.penalty(model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
            ).float()
            loss_hist_forget[epoch] += loss.item() * y_batch.size(0)
            accuracy_hist_forget[epoch] += is_correct.sum()
            entropy_hist_forget[epoch] += entropy(pred)
            confidence_hist_forget[epoch] += confidence(pred)

        accuracy_hist_forget[epoch] /= len(forget_dl.dataset)
        loss_hist_forget[epoch] /= len(forget_dl.dataset)
        entropy_hist_forget[epoch] /= len(forget_dl.dataset)
        confidence_hist_forget[epoch] /= len(forget_dl.dataset)
        
        model.eval()


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


            for x_batch, y_batch in train_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_train[epoch] += \
                    loss.item() * y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_train[epoch] += is_correct.sum()
                entropy_hist_train[epoch] += entropy(pred)
                confidence_hist_train[epoch] += confidence(pred)


            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
            entropy_hist_valid[epoch] /= len(valid_dl.dataset)
            confidence_hist_valid[epoch] /= len(valid_dl.dataset)

            loss_hist_train[epoch] /= len(train_dl.dataset)
            accuracy_hist_train[epoch] /= len(train_dl.dataset)
            entropy_hist_train[epoch] /= len(train_dl.dataset)
            confidence_hist_train[epoch] /= len(train_dl.dataset)

            print(f'----- Epoch {epoch+1} -----\n'
                  f' Forget Acc: {accuracy_hist_forget[epoch]:.4f}\n'
                  f' Forget Loss: {loss_hist_forget[epoch]:.4f}\n'
                  f' Forget Entropy: {entropy_hist_forget[epoch]:.4f}\n'
                  f' Forget Confidence: {confidence_hist_forget[epoch]:.4f}\n\n'
                  
                  f' Train Acc: {accuracy_hist_train[epoch]:.4f}\n'
                  f' Train Loss: {loss_hist_train[epoch]:.4f}\n'
                  f' Train Entropy: {entropy_hist_train[epoch]:.4f}\n'
                  f' Train Confidence: {confidence_hist_train[epoch]:.4f}\n\n'
                  
                  f' Valid Acc: {accuracy_hist_valid[epoch]:.4f}\n'
                  f' Valid Loss: {loss_hist_valid[epoch]:.4f}\n'
                  f' Valid Entropy: {entropy_hist_valid[epoch]:.4f}\n'
                  f' Valid Confidence: {confidence_hist_valid[epoch]:.4f}\n'
                  f'----------------------\n')
            
            wandb.log({"Forget Acc": accuracy_hist_forget[epoch], 
                       "Forget Loss": loss_hist_forget[epoch], 
                       "Forget Entropy": entropy_hist_forget[epoch],
                       "Forget Confidence": confidence_hist_forget[epoch], 

                       "Train Acc": accuracy_hist_train[epoch],
                       "Train Loss": loss_hist_train[epoch],
                       "Train Entropy": entropy_hist_train[epoch],
                       "Train Confidence": confidence_hist_train[epoch], 
            
                       "Valid Loss": loss_hist_valid[epoch],
                       "Valid Acc": accuracy_hist_valid[epoch],
                       "Valid Entropy": entropy_hist_valid[epoch],
                       "Valid Confidence": confidence_hist_valid[epoch], 
                    })

forget(model, 200, forget_dl)

# TODO:
# Basiline for Random Input & Output