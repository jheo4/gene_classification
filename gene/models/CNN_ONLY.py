import torch
import torch.nn as nn

class CNN_ONLY(nn.Module):
  def __init__(self):
    super(CNN_ONLY, self).__init__()
    self.c1 = nn.Conv2d(1, 36, kernel_size=3, padding=1)
    self.b1 = nn.BatchNorm2d(36)

    self.c2 = nn.Conv2d(36, 24, kernel_size=1)
    self.b2 = nn.BatchNorm2d(24)

    self.c3 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
    self.b3 = nn.BatchNorm2d(64)
    self.p1 = nn.MaxPool2d(kernel_size = 2, stride=2)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(1024, 3)

  def forward(self, x):
    #print(x.size())
    x = self.relu(self.b1(self.c1(x)))
    x = self.relu(self.b2(self.c2(x)))
    x = self.relu(self.b3(self.c3(x)))
    x = self.p1(x)
    x = x.view(x.size(0), -1)
    #print(x.size())
    x = self.linear(x)
    return x
