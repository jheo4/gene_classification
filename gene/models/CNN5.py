import torch
import torch.nn as nn

class CNN5(nn.Module):
  def __init__(self):
    super(CNN5, self).__init__()
    self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.b1 = nn.BatchNorm2d(16)
    self.c2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.b2 = nn.BatchNorm2d(16)
    self.p1 = nn.AvgPool2d(kernel_size = 2, stride=2)

    self.c3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.b3 = nn.BatchNorm2d(32)
    self.c4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.b4 = nn.BatchNorm2d(32)
    self.p2 = nn.AvgPool2d(kernel_size = 2, stride=2)

    self.relu = nn.ReLU()
    self.linear = nn.Linear(32*4, 3)

  def forward(self, x):
    #print(x.size())
    x = self.relu(self.b1(self.c1(x)))
    x = self.relu(self.b2(self.c2(x)))
    x = self.p1(x)

    #print(x.size())
    x = self.relu(self.b3(self.c3(x)))
    x = self.relu(self.b4(self.c4(x)))
    x = self.p2(x)
    x = x.view(x.size(0), -1)
    #print(x.size())
    x = self.linear(x)
    return x
