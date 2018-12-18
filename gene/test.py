import torch
import torch.nn as nn
m = nn.MaxPool1d(2, stride=2)
input = torch.randn(20)
output = m(input)

print(output.size())
