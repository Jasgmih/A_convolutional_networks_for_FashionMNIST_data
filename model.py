import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self): 
        super(CNNModel, self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.layernorm1 = nn.LayerNorm([24,24])
        
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.layernorm2 = nn.LayerNorm([20,20])

        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Linear(576, 10) 
        
    def forward(self, x):
        x = self.layernorm1(self.relu1(self.cnn1(x)))
        x = self.layernorm2(self.relu2(self.cnn2(x)))
        x = self.pool3(self.relu3(self.cnn3(x)))
        x = self.pool4(self.relu4(self.cnn4(x)))
        
        y = x.flatten(start_dim=1) 
        
        return self.linear(y)