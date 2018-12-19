import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data import gene_dataset as gd

# select models
from models import NN2
from models import NN3
from models import vgg3
from models import vgg5

batch_size = 256
trainset = gd.GeneDataset(train=True, dim=2)
testset = gd.GeneDataset(train=False, dim=2)

train_loader = torch.utils.data.DataLoader(dataset=trainset,
    batch_size=batch_size, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=testset,
    batch_size=batch_size, shuffle=False, num_workers=2)


# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = ('EI', 'IE', 'N')

# set model
net = vgg3.Net()
if torch.cuda.is_available():
  net.cuda()

loss_function = torch.nn.CrossEntropyLoss().cuda()  # cuda
learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,
    weight_decay=1e-4)
best_performacne = {'epoch':0, 'accuracy':0}

# Training
for epoch in range(0, 600):
  if epoch < 100:
    learning_rate = 0.01
  elif epoch < 300:
    learning_rate = 0.005
  else:
    learning_rate = 0.001
  for params in optimizer.param_groups:
    params['lr'] = learning_rate

  running_loss = 0.0
  total_batch = len(trainset) // batch_size

  net.train()
  for batch_index, (gene_code, label) in enumerate(train_loader):
    if torch.cuda.is_available():
      gene_code, label = Variable(gene_code.cuda()), Variable(label.cuda())
    else:
      gene_code, label = Variable(gene_code), Variable(label)

    optimizer.zero_grad()
    outputs = net(gene_code)
    loss = loss_function(outputs, label)
    loss.backward()
    optimizer.step()
    running_loss += loss / total_batch

  print("Epoch %d: Loss %.3f" % (epoch+1, running_loss))

  correct = 0
  total = 0

  net.eval()
  for batch_index, (gene_code, label) in enumerate(test_loader):
    if torch.cuda.is_available():
      gene_code, label = Variable(gene_code.cuda()), Variable(label.cuda())
    else:
      gene_code, label = Variable(gene_code), Variable(label)
    output = net(gene_code)
    _, predicted = torch.max(output.data, 1)
    total += label.size(0)
    correct += predicted.eq(label.data).cpu().sum()

  accuracy = (100 * correct / total)
  if best_performacne['accuracy'] < accuracy:
    best_performacne['epoch'] = epoch+1
    best_performacne['accuracy'] = accuracy
  print('Accuracy of epoch(%d): %d' % (epoch+1, accuracy))

print('The best performance: epoch %d/ accuracy %d' %\
    (best_performacne['epoch'], best_performacne['accuracy']))

