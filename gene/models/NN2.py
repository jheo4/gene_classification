import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.ln1 = nn.Linear(64, 40, bias=True)
    self.ln2 = nn.Linear(40, 3, bias=True)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.ln1(x))
    x = self.ln2(x)
    return x
