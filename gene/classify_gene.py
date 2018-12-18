import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data import gene_dataset as gd

# select models
from models import basic_NN as NN

batch_size = 128
trainset = gd.GeneDataset(train=True)
testset = gd.GeneDataset(train=False)

train_loader = torch.utils.data.DataLoader(dataset=trainset,
    batch_size=batch_size, shuffle=True, num_workers=2)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = ('EI', 'IE', 'N')

# set model
net = NN.Net()

loss_function = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
epochs = 400

# Training
for epoch in range(epochs):
  running_loss = 0.0
  total_batch = len(trainset) // batch_size
  for i, (gene_code, label) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = net(gene_code)
    loss = loss_function(outputs, label)
    loss.backward()
    optimizer.step()

    running_loss += loss / total_batch
  print("Epoch %d: Loss %.3f" % (epoch+1, running_loss))

  correct = 0
  total = 0
  for gene_code, label in testset:
    output = net(gene_code)
    _, predicted = torch.max(output.data, 0)
    total += 1
    correct += (predicted == label).sum()
  print('Accuracy of epoch(%d): %d' % (epoch+1, (100 * correct / total)))

