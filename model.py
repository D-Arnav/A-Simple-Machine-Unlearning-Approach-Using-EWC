import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.C2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.FC1 = nn.Linear(1568, 512)
        self.FC2 = nn.Linear(512, 10)
        self.ReLU = nn.ReLU()
        self.Pool = nn.MaxPool2d(kernel_size=2)
        self.Flatten = nn.Flatten()
        self.Dropout = nn.Dropout(p=0.5)
    
    def forward(self, input):
        x = self.C1(input)
        x = self.ReLU(x)
        x = self.Pool(x)
        x = self.C2(x)
        x = self.ReLU(x)
        x = self.Pool(x)
        x = self.Flatten(x)
        x = self.FC1(x)
        x = self.ReLU(x)
        x = self.Dropout(x)
        x = self.FC2(x)
        return x
    