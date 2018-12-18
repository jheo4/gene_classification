# Data info
# EI 0~766(767): Training(687) Testing(80)
# IE 767~1535(768): Training(688) Testing(80)
# N  1536~END(1655): Training(1490) Testing(165)
# Trainset(2865) / Testset(325)

import torch
from torch.utils.data import Dataset, DataLoader

class GeneDataset(Dataset):
  def __init__(self, train):
    if train == True:
      print("Prepare trainset...", end='   ')
      self.data, _ = self.prepare_data(1)
      self.len = 2865
    else:
      print("Prepare testset...", end='   ')
      _, self.data = self.prepare_data(1)
      self.len = 325
    print("[finished]")

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return self.len


  def prepare_data(self, dimension):
    raw_data_file = open("./data/splice.data", 'r')
    EIs = []
    IEs = []
    Ns = []
    EI_split = 687
    IE_split = 688
    N_split = 1490
    trainset = []
    testset = []

    while True:
      read_data = raw_data_file.readline()
      if not read_data:
        break
      gene_value = []
      data = []
      if dimension == 1:
        for i in range(39, 99):
          gene_value.append(ord(read_data[i]))
        tensor_value = torch.tensor(gene_value, dtype=torch.float)
        tensor_value = tensor_value / 255
        data.append(tensor_value)

        # value 0(EI) 1(IE) 2(N)
        label = read_data.split(',')[0]
        if label == 'EI':
          data.append(0)
          EIs.append(data)
        elif label == 'IE':
          data.append(1)
          IEs.append(data)
        elif label == 'N':
          data.append(2)
          Ns.append(data)
      elif dimension == 2: # remained for conv2d...
        pass
    raw_data_file.close()

    trainset += EIs[ :EI_split]
    testset += EIs[EI_split: ]

    trainset += IEs[ :IE_split]
    testset += IEs[IE_split: ]

    trainset += Ns[ :N_split]
    testset += Ns[N_split: ]

    return trainset, testset


if __name__ == '__main__':
  import torch
  import torchvision
  import torchvision.transforms as transforms
  trainset = GeneDataset(train=True)
  testset = GeneDataset(train=False)

  train_loader = DataLoader(dataset=trainset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=2)
  for i, data in enumerate(train_loader):
    gene_code, label = data
    print(gene_code.size())
