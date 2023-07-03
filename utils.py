import os
import shutil
import torch
import numpy as np
from torch.nn import functional as F
from scipy.spatial.distance import pdist, squareform
import numpy as np

import os
import time
# from catboost import Pool
import pandas as pd
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

def save_model_by_name(model_dir, model, global_step=None, history=None):
	save_dir = os.path.join('checkpoints', model_dir, model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	if history is not None:
		np.save(os.path.join(save_dir, 'test_metrics_history'), history)
	print('Saved to {}'.format(file_path))

def load_model_by_name(model_dir, model, global_step=None):
	"""
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
	file_path = os.path.join('checkpoints', model_dir, model.name,
							 'model.pt')
	state = torch.load(file_path)
	model.load_state_dict(state)
	print("Loaded from {}".format(file_path))

ce = torch.nn.CrossEntropyLoss(reduction='none')

def cross_entropy_loss(x, logits):
	"""
	Computes the log probability of a Bernoulli given its logits

	Args:
		x: tensor: (batch, dim): Observation
		logits: tensor: (batch, dim): Bernoulli logits

	Return:
		log_prob: tensor: (batch,): log probability of each sample
	"""
	log_prob = ce(input=logits, target=x).sum(-1)
	return log_prob



def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor