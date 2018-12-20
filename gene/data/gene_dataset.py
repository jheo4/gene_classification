# Data info
# EI 0~766(767): Training(687) Testing(80)
# IE 767~1535(768): Training(688) Testing(80)
# N  1536~END(1655): Training(1490) Testing(165)
# Trainset(2865) / Testset(325)

import torch
from torch.utils.data import Dataset, DataLoader


class GeneDataset(Dataset):
  def __init__(self, train=True, dim=2, classic=False):
    if classic == False:
      if train == True:
        print("Prepare trainset...", end='   ')
        self.data, _ = self.prepare_data(dim)
        self.len = 2865
      else:
        print("Prepare testset...", end='   ')
        _, self.data = self.prepare_data(dim)
        self.len = 325
    print("[finished]")


  def __getitem__(self, index):
    return self.data[index]


  def __len__(self):
    if classic == True:
      return 3190
    return self.len


  def augment_data(self):
    # D/N/S/R Augmentation
    # D : A, G, or T
    # N : A, G, C, or T
    # S : C or G
    # R : A or G
    for i in range(0, self.len):
      pass


  def prepare_data(self, dim):
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
      tensor_value = []

      # zero pading 00AT...TA00, 1x60 -> 1x64 (8x8 for CNN)
      gene_value.append(0)
      gene_value.append(0)
      for i in range(39, 99):
        if read_data[i] == 'A':
          gene_value.append(60)
        elif read_data[i] == 'T':
          gene_value.append(120)
        elif read_data[i] == 'G':
          gene_value.append(180)
        elif read_data[i] == 'C':
          gene_value.append(240)
        else: # for D/N/S/R
          gene_value.append(20)
          #gene_value.append(ord(read_data[i]))
      gene_value.append(0)
      gene_value.append(0)

      if dim == 1:
        tensor_value = gene_value
      elif dim == 2: # remained for conv2d...
        temp_value = []
        for i in range(0, 8):
          temp = gene_value[i*8:(i*8)+8]
          temp_value.append(temp)
        tensor_value.append(temp_value)

      tensor_value = torch.tensor(tensor_value, dtype=torch.float)
      tensor_value = tensor_value / 255
      data.append(tensor_value)

      # value 0(EI) 1(IE) 2(N)
      label = read_data.split(',')[0]
      if label == 'EI':
        data.append(torch.tensor(0))
        EIs.append(data)
      elif label == 'IE':
        data.append(torch.tensor(1))
        IEs.append(data)
      elif label == 'N':
        data.append(torch.tensor(2))
        Ns.append(data)

    raw_data_file.close()

    trainset += EIs[ :EI_split]
    testset += EIs[EI_split: ]

    trainset += IEs[ :IE_split]
    testset += IEs[IE_split: ]

    trainset += Ns[ :N_split]
    testset += Ns[N_split: ]

    return trainset, testset


  def prepare_calssic_data(self):
    raw_data_file = open("./data/splice.data", 'r')
    EIs = []
    EI_lables = []
    IEs = []
    IE_labels = []
    Ns = []
    N_labels = []
    EI_split = 687
    IE_split = 688
    N_split = 1490

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    while True:
      read_data = raw_data_file.readline()
      if not read_data:
        break
      gene_value = []

      for i in range(39, 99):
        if read_data[i] == 'A':
          gene_value.append(60)
        elif read_data[i] == 'T':
          gene_value.append(120)
        elif read_data[i] == 'G':
          gene_value.append(180)
        elif read_data[i] == 'C':
          gene_value.append(240)
        else: # for D/N/S/R
          gene_value.append(20)
          #gene_value.append(ord(read_data[i]))

      # value 0(EI) 1(IE) 2(N)
      label = read_data.split(',')[0]
      if label == 'EI':
        EIs.append(gene_value)
        EI_lables.append(0)
      elif label == 'IE':
        IEs.append(gene_value)
        IE_labels.append(1)
      elif label == 'N':
        Ns.append(gene_value)
        N_labels.append(2)

    raw_data_file.close()

    train_data += EIs[ :EI_split]
    test_data += EIs[EI_split: ]
    train_labels += EI_lables[ :EI_split]
    test_labels += EI_lables[EI_split: ]

    train_data += IEs[ :IE_split]
    test_data += IEs[IE_split: ]
    train_labels += IE_labels[ :IE_split]
    test_labels += IE_labels[IE_split: ]

    train_data += Ns[ :N_split]
    test_data += Ns[N_split: ]
    train_labels += N_labels[ :N_split]
    test_labels += N_labels[N_split: ]

    return train_data, train_labels, test_data, test_labels



if __name__ == '__main__':
  import torch
  import torchvision
  import torchvision.transforms as transforms
  trainset = GeneDataset(train=True, dim=2)
  testset = GeneDataset(train=False, dim=2)
  train_loader = DataLoader(dataset=trainset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=2)
  print(trainset[0])
  for i, data in enumerate(train_loader):
    gene_code, label = data
    #print(gene_code.size())
