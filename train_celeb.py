import os
import math
import time
import random
import torch
import argparse
import numpy as np
import torch
import torch.nn as nn
# import utils as ut
import torch.optim as optim
from torch import autograd
import utils as ut
from torch.utils import data
import torch.utils.data as Data
from torch.autograd import Variable
import dataloader as dataload
from codebase.method.config import get_config
from codebase.method.learner import Learner
import sklearn.metrics as skm
import warnings
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("torch.cuda.is_available():%s" % (torch.cuda.is_available()))

args, _ = get_config()
workstation_path = './'
print('Experiment: epsilon={},  bias={},  iv_lr={}'.format(args.epsilon, args.bias, args.iv_lr))
print('================================================')
# train_dataset_dir = os.path.join(workstation_path, 'dataset/', 'douban', 'dataset.npy')
# test_dataset_dir = os.path.join(workstation_path, 'dataset/', 'douban', 'dataset.npy')
if args.dataset == 'huawei':
    train_dataset_dir = os.path.join(workstation_path, 'dataset/', 'huawei', 'train.csv')
    test_dataset_dir = os.path.join(workstation_path, 'dataset/', 'huawei', 'train.csv')
    train_dataloader = dataload.dataload_huawei(train_dataset_dir, args.batch_size, mode='train')
    test_dataloader = dataload.dataload_huawei(test_dataset_dir, args.batch_size, mode='test')
elif args.dataset[:3] == 'non' or args.dataset == 'celeba': # the synthetic data drop here
    print(args)
    train_dataset_dir = os.path.join(workstation_path, 'dataset/', args.dataset, 'train', 'data_nonuniform.csv')
    test_dataset_dir = os.path.join(workstation_path, 'dataset/', args.dataset, 'dev', 'data_uniform.csv')
    train_dataloader = dataload.dataload_huawei(train_dataset_dir, args.batch_size, mode='train')
    test_dataloader = dataload.dataload_huawei(test_dataset_dir, args.batch_size, mode='test')
elif args.dataset in ['coat', 'yahoo', 'pcic']:
    train_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'data_nonuniform.csv')
    test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_uniform.csv')
    if args.dataset in ['coat']:
        data_feature_path = os.path.join(workstation_path, 'dataset', args.dataset)
    else:
        args.feature_data = False
        data_feature_path = None
    print(args)
    pretrain_dataloader = dataload.dataload_ori(train_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                                syn=False)
    train_dataloader = dataload.dataload_ori(train_dataset_dir, args.batch_size,
                                             data_feature_path=data_feature_path, syn=False)
    test_dataloader = dataload.dataload_ori(test_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                            syn=False)

# 构建模型
model = Learner(args)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
iv_optimizer = torch.optim.Adam(model.parameters(), lr=args.iv_lr, betas=(0.9, 0.999))

for epoch in range(args.epoch_max):
    model.train()
    total_auc = 0
    total_acc = 0
    total_logloss = 0
    total_ds_loss = 0
    total_kl = 0
    total_iv_loss = 0
    iv_loss = 0
    # for step in range(args.max_step):
    if args.mode == 'CausalRep':
        iv_count = 1
        for x, y in train_dataloader:
            optimizer.zero_grad()
            loss, ds_loss, kl, z, _ = model.learn(x, y)

            if args.downstream == 'MLP':
                auc = skm.roc_auc_score(y.cpu().numpy(),
                                        torch.argmax(model.predict(x)[0], dim=1).cpu().detach().numpy())
                acc = accuracy_score(y.cpu().numpy(), torch.argmax(model.predict(x)[0], dim=1).cpu().detach().numpy())
                logloss = log_loss(y.cpu().numpy(),
                                   F.sigmoid(torch.max(model.predict(x)[0], dim=1)[0]).cpu().detach().numpy())
            else:
                y_pred = model.predict(x)
                auc = skm.roc_auc_score(y.cpu().numpy(), y_pred.cpu().detach().numpy())
                acc = skm.accuracy_score(y.cpu().numpy(), torch.where(y_pred > 0.5, torch.ones_like(y_pred),
                                                                      torch.zeros_like(y_pred)).cpu().detach().numpy())
            L = loss.mean()  # + iv_loss.mean()
            L.backward()
            optimizer.step()
            total_auc += auc
            total_acc += acc
            total_logloss += logloss
            total_ds_loss += ds_loss
            total_kl += kl
            m = len(train_dataloader)
            # for x, y in train_dataloader:
            if args.mode == 'IB':
                continue
            iv_optimizer.zero_grad()
            iv_loss = model.learn_neg(x, y)
            # print()
            if iv_loss is not None:
                # iv_optimizer.zero_grad()
                # iv_loss = model.learn_neg(x, y)
                L = iv_loss.mean()
                L.backward()
                iv_optimizer.step()
                total_iv_loss += iv_loss
                iv_count += 1
            else:
                continue
    # test
    if epoch % 1 == 0:
        test_advauc = 0
        test_advacc = 0
        total_test_auc = 0
        total_test_acc = 0
        total_test_advauc = 0
        total_test_advacc = 0
        total_test_logloss = 0
        total_test_pa = 0
        total_test_dc = 0
        total_test_nd = 0
        total_test_mic = 0
        total_test_tic = 0
        for test_x, test_y in test_dataloader:
            test_a = x[:, args.user_dim:]
            # print(test_x)
            if args.downstream in ['MLP', 'bprBPR', 'mlpBPR', 'gmfBPR', 'NeuBPR']:
                # print(test_y)
                # test_y_pred = model.predict(test_x, test_a)
                test_y_pred, z = model.predict(test_x, test_a)
                test_pa = ut.distcorr(z.cpu().detach().numpy(), test_x[:, :2].cpu().detach().numpy())
                test_nd = ut.distcorr(z.cpu().detach().numpy(), test_x[:, 2:4].cpu().detach().numpy())
                test_dc = ut.distcorr(z.cpu().detach().numpy(), test_x[:, 6].cpu().detach().numpy())
                test_auc = skm.roc_auc_score(test_y.cpu().numpy(),
                                             torch.argmax(test_y_pred, dim=1).cpu().detach().numpy())
                test_acc = accuracy_score(test_y.cpu().numpy(),
                                          torch.argmax(test_y_pred, dim=1).cpu().detach().numpy())
                test_adv_y_pred = model.eval_adv(test_x, test_y)
                test_advauc = skm.roc_auc_score(test_y.cpu().numpy(),
                                                torch.argmax(test_adv_y_pred, dim=1).cpu().detach().numpy())
                test_advacc = accuracy_score(test_y.cpu().numpy(),
                                             torch.argmax(test_adv_y_pred, dim=1).cpu().detach().numpy())
            else:
                test_y_pred = model.predict(test_x, test_a)
                test_adv_y_pred = model.eval_adv(test_x, test_y, test_a)
                test_auc = skm.roc_auc_score(test_y.cpu().numpy(), test_y_pred.cpu().detach().numpy())
                test_acc = skm.accuracy_score(test_y.cpu().numpy(),
                                              torch.where(test_y_pred > 0.5, torch.ones_like(test_y_pred),
                                                          torch.zeros_like(
                                                              test_y_pred)).cpu().detach().numpy())
                test_advauc = skm.roc_auc_score(test_y.cpu().numpy(), test_adv_y_pred.cpu().detach().numpy())
                test_advacc = skm.accuracy_score(test_y.cpu().numpy(),
                                                 torch.where(test_adv_y_pred > 0.5, torch.ones_like(test_adv_y_pred),
                                                             torch.zeros_like(
                                                                 test_adv_y_pred)).cpu().detach().numpy())
            total_test_auc += test_auc
            total_test_acc += test_acc
            total_test_pa += test_pa
            total_test_nd += test_nd
            total_test_dc += test_dc
            total_test_advauc += test_advauc
            total_test_advacc += test_advacc
        test_dataloader_len = len(test_dataloader)
        train_dataloader_len = len(train_dataloader)
        print("Epoch:{}\n test_auc:{}, test_acc:{}, test_pa:{}, test_nd:{}, test_adv_auc:{}, test_adv_acc:{}".format(epoch, float(
            total_test_auc / test_dataloader_len), float(total_test_acc / test_dataloader_len), float(total_test_pa / test_dataloader_len), float(total_test_nd / test_dataloader_len), float(total_test_dc / test_dataloader_len),
            float(total_test_advauc / test_dataloader_len), float(total_test_advacc / test_dataloader_len)))
    