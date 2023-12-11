import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import Subset, DataLoader, RandomSampler
import torchvision
from torchvision import transforms, datasets
from copy import deepcopy
import random

def get_loaders():
    image_path = './'
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist_dataset = torchvision.datasets.MNIST(
        root=image_path, train=True, transform=transform, download=True
    )

    mnist_valid_dataset = Subset(mnist_dataset, torch.arange(1000))
    mnist_train_dataset = Subset(mnist_dataset, torch.arange(1000, len(mnist_dataset)))
    mnist_forget_dataset = Subset(mnist_train_dataset, torch.arange(1000, 1048))

    batch_size = 16

    train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
    valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)
    forget_dl = DataLoader(mnist_forget_dataset, batch_size, shuffle=True)

    return train_dl, valid_dl, forget_dl

def entropy(y_hat):
    probas = nn.functional.softmax(y_hat, dim=1)
    entropy = -torch.sum(probas * torch.log(probas + 1e-10), dim=1)
    mean_entropy = torch.sum(entropy)
    return mean_entropy

def confidence(y_hat):
    probas = nn.functional.softmax(y_hat, dim=1)
    confidence, _ = torch.max(probas, dim=1)
    mean_confidence = torch.sum(confidence)
    return mean_confidence

def accuracy(y, y_hat):
    is_correct = (
        torch.argmax(y_hat, dim=1) == y
    ).float()
    return is_correct.mean()

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
    