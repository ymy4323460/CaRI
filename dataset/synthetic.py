import numpy as np
import pandas as pd
import os

def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))

def get_labels(pay, epsilon1, mode='nonlinear', beta=0.3):
    epsilon1 = np.random.randn(pay.shape[0],pay.shape[1]) + 0.3
    if mode == 'nonlinear':
        all_x = pay + beta * epsilon1
        indicate = np.where(all_x > 0, 1, 0)
        indicate_neg = np.where(all_x < 0, 1, 0)
        pos = np.multiply(all_x, indicate) - 0.5 * indicate
        neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
        result = (1 / 2.) * pos + (1 / 2.) * np.multiply(pos, neg)
        return np.where(sigmoid_fun(result.sum(1)) > 0.5, 1, 0), result
    else:
        all_x = pay + beta * epsilon1
        indicate = np.where(all_x > 0, 1, 0)
        indicate_neg = np.where(all_x < 0, 1, 0)
        pos = np.multiply(all_x, indicate) - 0.5 * indicate
        neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
        result = (1 / 2.) * pos + (1 / 4.) * (pos + neg)
        return np.where(sigmoid_fun(result.sum(1)) > 0.5, 1, 0), result


def get_dcy(y, ndy=None, beta=0.3, mode='nonlinear', dim=5):
    epsilon2 = np.random.randn(y.shape[0], dim) + 0.3
    if mode == 'nonlinear':
        all_x = y.reshape(y.shape[0], 5) + beta * epsilon2
        indicate = np.where(all_x > 0, 1, 0)
        indicate_neg = np.where(all_x < 0, 1, 0)
        pos = np.multiply(all_x, indicate) - 0.5 * indicate
        neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
        result = (1 / 2.) * pos + (1 / 2.) * np.multiply(pos, neg)
        return np.where(sigmoid_fun(result) > 0.5, 1, 0)
    else:
        all_x = y.reshape(y.shape[0], 1) + beta * epsilon2
        indicate = np.where(all_x > 0, 1, 0)
        indicate_neg = np.where(all_x < 0, 1, 0)
        pos = np.multiply(all_x, indicate) - 0.5 * indicate
        neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
        result = (1 / 2.) * pos + (1 / 4.) * (pos + neg)
        return result


def get_ndy(pay, epsilon3, beta=0.3, mode='nonlinear'):
    epsilon3 = np.random.randn(pay.shape[0], pay.shape[1])+0.6
    if mode == 'nonlinear':
        all_x = pay + beta * epsilon3
        indicate = np.where(all_x > 0, 1, 0)
        indicate_neg = np.where(all_x < 0, 1, 0)
        pos = np.multiply(all_x, indicate) - 0.5 * indicate
        neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
        result = (1 / 2.) * pos + (1 / 2.) * np.multiply(pos, neg)
        return result
    else:
        all_x = pay + beta * epsilon3
        indicate = np.where(all_x > 0, 1, 0)
        indicate_neg = np.where(all_x < 0, 1, 0)
        pos = np.multiply(all_x, indicate) - 0.5 * indicate
        neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
        result = (1 / 2.) * pos + (1 / 4.) * (pos + neg)
        return result

def data_generator(dimension, number=2000, beta=0.1):
    if not os.path.exists('./nonliearlogitdata_{}_{}/'.format(number, beta)):  #
        os.makedirs('./nonliearlogitdata_{}_{}/dev/'.format(number, beta))
        os.makedirs('./nonliearlogitdata_{}_{}/train/'.format(number, beta))

    pay = np.random.uniform(-1, 1, (number, dimension))
    y, z = get_labels(pay, beta)
    ndy = get_ndy(pay, beta)
    dcy = get_dcy(z, beta, dim=dimension)

    x = np.concatenate((pay, ndy, dcy), axis=1)
#     per_idx = np.random.permutation(dimension*3)
#     x = x[:, per_idx]
    x = np.concatenate((x, y.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    all_ = pd.DataFrame(x)

    partition = int(0.75 * all_.shape[0])
    pd.DataFrame(all_[: partition]).to_csv(
        './nonliearlogitdata_{}_{}/train/data_nonuniform.csv'.format(number, beta),
        header=False, index=False)
    pd.DataFrame(all_[partition:]).to_csv(
        './nonliearlogitdata_{}_{}/dev/data_uniform.csv'.format(number, beta),
        header=False, index=False)

for i in [0.1, 0.3, 0.5, 0.7, 1, 3]:
    data_generator(5, number=500, beta=i)

