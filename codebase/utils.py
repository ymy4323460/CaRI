import os
import shutil
import torch
import numpy as np
from torch.nn import functional as F
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples//2
    N = len(x.shape) - 1
    with torch.no_grad():
        # Q * B * C * H * W
        extender = [1]*N
        queries = x.repeat(Q, *extender)
        noise = torch.randn_like(queries)
        norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
        noise = noise / norm
        noise = torch.cat([-noise, noise])
        queries = torch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender)
        grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
    return grad


def project(x, orig_input, epsilon=0.01, norm=2):
	"""
    """
	diff = x - orig_input
	if norm == '2':
		diff = diff.renorm(p=2, dim=0, maxnorm=epsilon)
	elif norm == 'inf':
		diff = torch.clamp(diff, -epsilon, epsilon)
	return torch.clamp(orig_input + diff, -0.5, 0.5)


def step(x, g, step_size=0.001):
	"""
    """
	# Scale g so that each element of the batch is at least norm 1
	l = len(x.shape) - 1
	g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
	scaled_g = g / (g_norm + 1e-10)
	return x + scaled_g * step_size


# def random_perturb(x):
# 	"""
#     """
# 	print('hahhahhah')
# 	new_x = x + (torch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.eps)
# 	return torch.clamp(new_x, 0, 1)


def replace_out_ball(z, z_raw, epsilon=0.3):
	while np.abs(z - z_raw) < epsilon:
		z = np.random.randn(z_raw.shape)
	return z

def condition_prior(scale, label, dim):
	mean = torch.ones(label.size()[0], dim)
	var = torch.ones(label.size()[0], dim)
	# print(label)
	for i in range(label.size()[0]):
		# print(label[i])
		mul = (float(label[i])-scale[0])/(scale[1]-0)
		mean[i] = torch.ones(dim)*mul
		var[i] = torch.ones(dim)*1
	return mean.to(device), var.to(device)

def random_perturb_out_ball(x, y, bias=1, epsilon=0.01):
	# print(x.size())
	'''
	# replace_min = y == 0
	# # print(x.size())
	# x[replace_min] = 0.1*torch.rand_like(x[replace_min])+epsilon
	# replace_max = x > 0
	# x[replace_max] = 0.1*torch.rand_like(x[replace_max])
	'''
	y = y.reshape(x.size()[0], 1).repeat(1, x.size()[1])
	x = torch.where(y == 0, epsilon * torch.rand_like(x)+bias, epsilon * torch.rand_like(x))
	# replace_min = x < 0
	# x[replace_min] = x - epsilon - torch.abs(torch.rand_like(x))
	# replace_max = x > 0
	# x[replace_max] = x + epsilon + torch.abs(torch.rand_like(x))
	# new_x = np.random.randn(x.size()[0], )
	# new_z = torch.from_numpy(np.array(list(map(replace_out_ball, new_x, x))))
	return x

def random_perturb(x, epsilon=0.01):
	new_x = x + torch.randn(x.size()).to(device) * epsilon
	return new_x

def sample_gaussian(m,v):
	# reparameterization
	sample = torch.randn(m.size()).to(device)
	z = m + (v**0.5) * sample
	return z

def gaussian_parameters(h, dim=-1):
	"""
	Converts generic real-valued representations into mean and variance
	parameters of a Gaussian distribution

	Args:
		h: tensor: (batch, ..., dim, ...): Arbitrary tensor
		dim: int: (): Dimension along which to split the tensor for mean and
			variance

	Returns:z
		m: tensor: (batch, ..., dim / 2, ...): Mean
		v: tensor: (batch, ..., dim / 2, ...): Variance
	"""
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

def kl_normal(qm, qv, pm, pv):
	"""
	Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
	sum over the last dimension

	Args:
		qm: tensor: (batch, dim): q mean
		qv: tensor: (batch, dim): q variance
		pm: tensor: (batch, dim): p mean
		pv: tensor: (batch, dim): p variance

	Return:
		kl: tensor: (batch,): kl between each sample
	"""
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl


def save_model_by_name(model_dir, model, global_step, history=None):
	save_dir = os.path.join('checkpoints', model_dir, model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	if history is not None:
		np.save(os.path.join(save_dir, 'test_metrics_history'), history)
	print('Saved to {}'.format(file_path))

def load_model_by_name(model, global_step):
	"""
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
	file_path = os.path.join('checkpoints', model_dir, train_mode, model.name,
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

