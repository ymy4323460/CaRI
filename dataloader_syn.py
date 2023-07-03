import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from random import randint, sample

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class DataLoad_huawei(data.Dataset):
    def __init__(self, root, mode):
        # self.user_item = fileread(os.path.join(root,'dataset'))
        # self.user_item = fileread(root)

        # self.user_item = np.abs(np.random.rand(10000, 600))
        self.data_df = pd.read_csv(root, header=None).values
        print(self.data_df.shape)
        self.mode = mode
#         partition = int(0.75 * self.data_df.shape[0])
#         if self.mode == 'train':
#             self.data_df = self.data_df[:partition]
#         elif self.mode == 'test':
#             self.data_df = self.data_df[partition:]


        self.transforms = transforms.Compose([transforms.ToTensor()])
        # self.data_feature_path = data_feature_path

    def __getitem__(self, idx):
        # print(idx) 300dim x
        data = torch.from_numpy(self.data_df[idx])
        # user_fea = torch.from_numpy(self.data_df[idx, : -2])
        # item_fea = torch.from_numpy(self.data_df[idx: 42: 153])
        # match = torch.from_numpy(self.data_df[idx, 153:-2])
        # impression = torch.tensor(self.data_df[idx, -2])
        # click = torch.tensor(self.data_df[idx, -1])
        # print(self.data_df[idx])
#         return data[253:282], data[-1]
        return data[:-2], data[-2]

        # data = torch.from_numpy(self.user_item[idx])
        # impression = torch.from_numpy(np.ones(1))
        # if idx % 2 == 1:
        #     item = torch.from_numpy(np.ones(1))
        # else:
        #     item = torch.from_numpy(np.zeros(1))
        #
        # # numerical_feature, item_id, label, impression
        # return data[4:304], item, item, impression


    def __len__(self):
        return self.data_df.shape[0]


def dataload_huawei(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, mode='train'):
    if batch_size == 'all':
        if mode == 'train':
            batch_size = 75000
        else:
            batch_size = 25000
    dataset = DataLoad_huawei(dataset_dir, mode)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return dataset

def fileread(data_dir):
    print(data_dir)
    with open(data_dir, 'rb') as f:
        reader = f.readlines()
        data = []
        for line in reader:
            data.append(line.decode().strip().split(','))
    # print(len(data))
    data=np.array(data,dtype=float)
    return data#.reshape([len(data),4,1])

class DataLoad_ori(data.Dataset):
    def __init__(self, root, data_feature_path=None, syn=False):
        self.user_item = fileread(root)
        print(self.user_item.shape)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.data_feature_path = data_feature_path

        if data_feature_path is not None:
            if syn:
                self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'), allow_pickle=True).item()
                self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'), allow_pickle=True).item()
            else:
                self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'), allow_pickle=True)#.item()
                self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'), allow_pickle=True)#.item()

    def __getitem__(self, idx):
        # print(idx)
        data = torch.from_numpy(self.user_item[idx])
        if self.data_feature_path is not None:
            user = torch.from_numpy(self.userfeature[int(data[0].numpy())])
            item = torch.from_numpy(self.itemfeature[int(data[1].numpy())])
            context = torch.cat((user, item), dim=0)
            return context, data[2].int()
        else:
            return data[:2], data[2]

    def __len__(self):
        # print(len(self.user_item))
        return len(self.user_item)


def dataload_ori(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, data_feature_path=None, syn=False):
	dataset = DataLoad_ori(dataset_dir, data_feature_path=data_feature_path, syn=syn)
	dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
	return dataset

if __name__ == '__main__':
    dataload_huawei('./dataset/part_0_test')
