import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

logged_itter = 5

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (torch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        # Scale g so that each element of the batch is at least norm 1
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (torch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.eps)
        return torch.clamp(new_x, 0, 1)


# def get_adv_examples(x):
#             # Random start (to escape certain types of gradient masking)
#             if random_start:
#                 x = step.random_perturb(x)
#
#             iterator = range(iterations)
#             if do_tqdm: iterator = tqdm(iterator)
#
#             # Keep track of the "best" (worst-case) loss and its
#             # corresponding input
#             best_loss = None
#             best_x = None
#
#             # A function that updates the best loss and best input
#             def replace_best(loss, bloss, x, bx):
#                 if bloss is None:
#                     bx = x.clone().detach()
#                     bloss = losses.clone().detach()
#                 else:
#                     replace = m * bloss < m * loss
#                     bx[replace] = x[replace].clone().detach()
#                     bloss[replace] = loss[replace]
#
#                 return bloss, bx
#
#             # PGD iterates
#             for _ in iterator:
#                 x = x.clone().detach().requires_grad_(True)
#                 losses, out = calc_loss(step.to_image(x), target)
#                 assert losses.shape[0] == x.shape[0], \
#                         'Shape of losses must match input!'
#
#                 loss = ch.mean(losses)
#
#                 if step.use_grad:
#                     if est_grad is None:
#                         grad, = ch.autograd.grad(m * loss, [x])
#                     else:
#                         f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
#                         grad = helpers.calc_est_grad(f, x, target, *est_grad)
#                 else:
#                     grad = None
#
#                 with ch.no_grad():
#                     args = [losses, best_loss, x, best_x]
#                     best_loss, best_x = replace_best(*args) if use_best else (losses, x)
#
#                     x = step.step(x, grad)
#                     x = step.project(x)
#                     if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))


class InterventionVunerable(nn.Module):
    def __init__(self, args, decoder):
        super().__init__()
        self.name = 'IV'
        self.args = args

        self.decoder = decoder
        # if self.args.adversarial_type == 'adversarial':
        # 	self.step = L2Step(orig_input, self.args.epsilon, step_size, use_grad=True)

    def random_intervene(self, z, y):
        # raw_loss = self.decoder.loss(z, y)
        # z_new = z.data
        z_neg = z.detach()
        print(z_neg)
        # 	ut.random_perturb(z)
        # neg_loss = self.decoder.loss(z_neg, y)
        # replace = raw_loss < neg_loss
        # z_new[replace] = z_neg[replace]
        return z_neg, self.decoder.loss(z_neg.detach(), y)


    def iv(self, z, y):
        # backward
        # z_neg, neg_loss = self.random_intervene(z, y)
        loss = self.decoder.loss(z, y)
        return loss #torch.abs(neg_loss.detach() - loss)

class InformationBottleneck(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.name = 'IB'
        self.args = args

        self.encoder = encoder
        self.decoder = decoder

    # def adverserial_attack(self):


    def random_intervene(self, z, y=None):
        raw_loss = F.cross_entropy(self.decoder.predict(z), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        z_new = z.detach()
        z_neg = ut.random_perturb(z, epsilon=self.args.epsilon)
        neg_loss = F.cross_entropy(self.decoder.predict(z_neg), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        # replace = raw_loss < neg_loss
        # z_new[replace] = z_neg[replace]
        # print(z_neg[replace].size())
        z_new = torch.where(raw_loss < neg_loss, z_neg, z)
        return z_new

    def outball(self, z, y=None):
        raw_loss = F.cross_entropy(self.decoder.predict(z), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        z_new = z.detach()
        # if self.args.
        z_neg = ut.random_perturb_out_ball(z, y, bias=self.args.bias, epsilon=self.args.epsilon)
        neg_loss = F.cross_entropy(self.decoder.predict(z_neg), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        # replace = raw_loss > neg_loss
        # print(replace)
        # z_new1 = z_neg[replace]
        # y_new = y[replace]
        z_new1 = z_new[torch.where(raw_loss > neg_loss)].reshape(-1, z.size()[1])
        y_new = y[torch.where(raw_loss[:, 0] > neg_loss[:, 0])]
        return z_new1, y_new

    def inference_z(self, x):
        # forward
        m, v = self.encoder.predict(x)
        return m, v

    def generate_y(self, z):
        # forward
        y = self.decoder.predict(z)
        return y

    def neg_loss(self, x, y):
        if self.args.prior_type == 'conditional':
            kl, m, v = self.encoder.kl(x, y)
        else:
            kl, m, v = self.encoder.kl(x)
        # v = torch.ones_like(m)*0.0001
        z = ut.sample_gaussian(m, v)
        z_out, y = self.outball(z, y)
        # print(z_out.size(), y)
        if z_out.size()[0] == 0:
            return None
        # out_loss = -F.cross_entropy(self.decoder.predict(z_out), y)
        # print(z_out.size())
        out_loss = -F.cross_entropy(self.decoder.predict(z_out), y)

        return out_loss.mean()

    def loss(self, x, y, a=None):
        # backward
        if self.args.prior_type == 'conditional':
            kl, m, v = self.encoder.kl(x, y)
        else:
            kl, m, v = self.encoder.kl(x)
        # v = torch.ones_like(m)*0.0001
        z = ut.sample_gaussian(m, v)
        z_neg = self.random_intervene(z, y)
        # z_out = self.outball(z, y)

        ds_loss = self.decoder.loss(z_neg, y)
        # out_loss = -F.cross_entropy(self.decoder.predict(z_out), y)

        z_neg_loss = torch.zeros_like(ds_loss)
        return ds_loss.mean() + self.args.beta * kl.mean(), ds_loss, kl, z, z_neg_loss


class Encoder(nn.Module): # q(z|x)
    def __init__(self, args):
        super().__init__()
        self.name = 'Encoder'
        self.args = args

        self.encode = nn.Sequential(
            nn.Linear(self.args.x_dim, self.args.enc_layer_dims[0]),
            nn.ELU(),
            nn.Linear(self.args.enc_layer_dims[0], self.args.enc_layer_dims[1]),
            nn.ELU(),
            nn.Linear(self.args.enc_layer_dims[1], self.args.embedding_dim * 2)
        )

    def predict(self, x):
        h = self.encode(x)

        m, v = ut.gaussian_parameters(h, dim=1)
        v = torch.zeros_like(m)
        return m, v

    def kl(self, x, y=None):
        m, v = self.predict(x)


        if self.args.prior_type == 'conditional':
            pm, pv = ut.condition_prior([0, 1], y, self.args.embedding_dim)
        else:
            pm, pv = torch.zeros_like(m), torch.ones_like(m)
        # print(m)
        return ut.kl_normal(m, pv*0.0001, pm, pv*0.0001), m, v

class Decoder(nn.Module): # p(y|z)
    def __init__(self, args):
        super().__init__()
        self.name = 'Decoder'
        self.args = args

        self.decode = nn.Sequential(
            nn.Linear(self.args.embedding_dim, self.args.dec_layer_dims[0]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[0], self.args.dec_layer_dims[1]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[1], 2)
        )

        self.sigmd = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss()
        self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([self.args.class_weight[0], self.args.class_weight[1]])).float(), reduction='none')

    def predict(self, z):
        return self.decode(z)

    def loss(self, z, y):
        return self.sftcross(self.predict(z), y.squeeze_())


class CTRModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = 'Decoder'
        self.args = args

        self.decode = nn.Sequential(
            nn.Linear(self.args.x_dim, self.args.dec_layer_dims[0]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[0], self.args.dec_layer_dims[1]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[1], 2)
        )

        self.sigmd = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss()
        self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([self.args.class_weight[0], self.args.class_weight[1]])).float(), reduction='none')

    def predict(self, z):
        return self.decode(z)

    def loss(self, z, y):
        return self.sftcross(self.predict(z), y.squeeze_())


class NeuBPR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.layers = [int(l) for l in args.layers.split('|')]
        # self.layers = args.layers
        if args.user_dim == 1:
            self.W_mlp = torch.nn.Embedding(num_embeddings=args.user_item_size[0], embedding_dim=args.user_emb_dim)
            self.W_mf = torch.nn.Embedding(num_embeddings=args.user_item_size[0], embedding_dim=args.user_emb_dim)
        else:
            self.W_mlp = torch.nn.Linear(self.args.user_dim, self.args.user_emb_dim)
            self.W_mf = torch.nn.Linear(self.args.user_dim, self.args.user_emb_dim)
        if args.item_dim == 1:
            self.H_mlp = torch.nn.Embedding(num_embeddings=args.user_item_size[1], embedding_dim=args.item_emb_dim)
            self.H_mf = torch.nn.Embedding(num_embeddings=args.user_item_size[1], embedding_dim=args.item_emb_dim)
        else:
            self.H_mlp = torch.nn.Linear(self.args.item_dim, self.args.item_emb_dim)
            self.H_mf = torch.nn.Linear(self.args.item_dim, self.args.item_emb_dim)

        nn.init.xavier_normal_(self.W_mlp.weight.data)
        nn.init.xavier_normal_(self.H_mlp.weight.data)
        nn.init.xavier_normal_(self.W_mf.weight.data)
        nn.init.xavier_normal_(self.H_mf.weight.data)

        if self.args.downstream == 'NeuBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.args.ctr_layer_dims[:-1], self.args.ctr_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.ctr_layer_dims[-1] + args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'gmfBPR':
            self.affine_output = torch.nn.Linear(in_features=args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'mlpBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.args.ctr_layer_dims[:-1], self.args.ctr_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.ctr_layer_dims[-1], out_features=1)

        self.logistic = torch.nn.Sigmoid()
        self.weight_decay = args.weight_decay
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.sftcross = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.args.ctr_classweight[1]))


    def loss(self, u, i, y):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:s
            torch.FloatTensor
        """
        # print(u.size(), i.size())
        if self.args.is_debias:
            u = u.reshape(u.size()[0], self.args.user_emb_dim)
        else:
            if self.args.user_dim == 1:
                u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.size()[0], 1)
            else:
                u = torch.tensor(u, dtype=torch.float32).to(device).reshape([u.size()[0], self.args.user_dim])
        if self.args.item_dim == 1:
            i = torch.tensor(i, dtype=torch.int64).to(device).reshape([i.size()[0], self.args.item_dim])
        else:
            i = torch.tensor(i, dtype=torch.float32).to(device).reshape([i.size()[0], self.args.item_dim])
        y = torch.tensor(y, dtype=torch.float32).to(device)
        x_ui = self.predict(u, i, mode='train')
        # x_uj = self.predict(u, j, mode='train')
        # x_uij = x_ui - x_uj
        # -------------------------------Mengyue Yang---------------------------------
        # # log_prob = F.logsigmoid(x_uij).mean()
        # log_prob = F.logsigmoid(x_uij)
        if not self.args.is_debias:
            Wu_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            Wu_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)

        Hi_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
        Hi_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)


        # print('***'*10)
        # print('x_uij', x_uij.size(), Wu_mlp.size(), Hi_mlp.size())
        # print('***'*10)

        # log_prob = F.logsigmoid(x_uij).mean()

        # if self.args.model_name == 'NeuBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Wu_mf.norm(dim=1).pow(2).mean() + Hi_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name in ['gmfBPR', 'bprBPR']:
        # 	regularization = self.weight_decay * (Wu_mf.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name == 'mlpBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mlp.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean())

        log_prob = self.sftcross(x_ui, y)

        # -----------------------------------------------------------------------
        if self.args.downstream == 'NeuBPR':
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mlp.norm(dim=1) + Hi_mf.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Wu_mf.norm(dim=1) + Hi_mlp.norm(dim=1) + Hi_mf.norm(dim=1))
        elif self.args.downstream in ['gmfBPR', 'bprBPR']:
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mf.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mf.norm(dim=1) + Hi_mf.norm(dim=1))
        elif self.args.downstream == 'mlpBPR':
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mlp.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Hi_mlp.norm(dim=1))
        # ------------------------------------------------------------------------
        return (log_prob + regularization).mean()

    # ----------------------------Quanyu Dai----------------------------------
    def predict(self, u, i, mode='test'):
        #
        # if mode == 'test':
        #     u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.shape[0], 1)
        #     i = torch.tensor(i, dtype=torch.int64).to(device).reshape(i.shape[0], 1)

        if self.args.downstream == 'NeuBPR':
            if self.args.is_debias:
                user_embedding_mlp = u.reshape(u.size()[0], self.args.user_emb_dim)
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        elif self.args.downstream == 'gmfBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.downstream == 'bprBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.downstream == 'mlpBPR':
            if self.args.is_debias:
                user_embedding_mlp = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector

            for idx, _ in enumerate(range(len(self.fc_layers))):
                vector = self.fc_layers[idx](vector)
                vector = torch.nn.ReLU()(vector)
                vector = self.dropout(vector)



        # print('###'*10)
        # print('user_emb, item_emb, vector', user_embedding_mf.size(), item_embedding_mf.size(), vector.size())
        # print('###'*10)

        if self.args.downstream in ['NeuBPR', 'gmfBPR', 'mlpBPR']:
            logits = self.affine_output(vector)
            rating = logits.reshape(logits.size()[0])
        elif self.args.downstream == 'bprBPR':
            rating = vector.sum(dim=1)
            rating = rating.reshape(rating.size()[0])

        if mode == 'test':
            # rating = self.logistic(rating)
            rating = rating#.detach().cpu().numpy()

        # print('rating', rating.shape, rating)

        return rating


class CrossLayer(nn.Module):
    def __init__(self, x_dim):
        super(CrossLayer, self).__init__()
        self.x_dim = x_dim
        self.weights = nn.Parameter(torch.zeros(x_dim, 1))  # x_dim * 1
        nn.init.xavier_uniform_(self.weights.data)
        self.bias = nn.Parameter(torch.randn(x_dim))  # x_dim

    def forward(self, x0, xi):
        # x0,x1: N * x_dim
        # print(x0.size(), xi.size())
        x = torch.mm(xi, self.weights)  # N * x_dim
        x = torch.sum(x, dim=1)  # N
        # x = x.unsqueeze(dim=1)  # N * 1
        # print(x.size())
        x = torch.matmul(x, x0)  # N * x_dim
        x = x + self.bias + xi
        return x




class MLP(nn.Module):

    def __init__(self, fc_in_dim, fc_dims, dropout=None, batch_norm=None, activation=nn.ReLU()):
        """
        The MLP(Multi-Layer Perceptrons) module
        :param fc_in_dim: The dimension of input tensor
        :param fc_dims: The num_neurons of each layer, should be array-like
        :param dropout: The dropout rate of the MLP module, can be number or array-like ranges (0,1), by default None
        :param batch_norm: Whether to use batch normalization after each layer, by default None
        :param activation: The activation function used in each layer, by default nn.ReLU()
        """
        super(MLP, self).__init__()
        self.fc_dims = fc_dims
        layer_dims = [fc_in_dim]
        layer_dims.extend(fc_dims)
        layers = []

        if not dropout:
            dropout = np.repeat(0, len(fc_dims))
        if isinstance(dropout, float):
            dropout = np.repeat(dropout, len(fc_dims))

        for i in range(len(layer_dims) - 1):
            fc_layer = nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1])
            nn.init.xavier_uniform_(fc_layer.weight)
            layers.append(fc_layer)
            if batch_norm:
                batch_norm_layer = nn.BatchNorm1d(num_features=layer_dims[i + 1])
                layers.append(batch_norm_layer)
            layers.append(activation)
            if dropout[i]:
                dropout_layer = nn.Dropout(dropout[i])
                layers.append(dropout_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature):
        y = self.mlp(feature)
        return y


class DCN(nn.Module):
    def __init__(self, args, fc_dims=None,
                 dropout=None, batch_norm=None):
        super(DCN, self).__init__()
        self.args = args
        # deep network
        if not fc_dims:
            fc_dims = [self.args.enc_layer_dims[0], self.args.enc_layer_dims[1]]
        self.fc_dims = fc_dims
        self.deep = MLP(self.args.x_dim, self.args.enc_layer_dims[1], dropout, batch_norm)
        # cross network
        cross_layers = []
        for _ in range(self.args.cross_depth):
            cross_layers.append(CrossLayer(self.args.x_dim))
        self.cross = nn.ModuleList(cross_layers)
        self.out_layer = MLP(self.args.enc_layer_dims[1]+self.args.x_dim, self.args.embedding_dim * 2)


    def embeddings(self, continuous_value):
        x0 = continuous_value
        y_dnn = self.deep(x0)

        xi = x0
        for cross_layer in self.cross:
            xi = cross_layer(x0, xi)

        output = torch.cat([y_dnn, xi], dim=1)
        output = self.out_layer(output)
        return output

    def predict(self, x):
        h = self.embeddings(x)

        m, v = ut.gaussian_parameters(h, dim=1)
        v = torch.zeros_like(m)
        return m, v

    def kl(self, x, y=None):
        m, v = self.predict(x)

        if self.args.prior_type == 'conditional':
            pm, pv = ut.condition_prior([0, 1], y, self.args.embedding_dim)
        else:
            pm, pv = torch.zeros_like(m), torch.ones_like(m)
        # print(m)
        return ut.kl_normal(m, pv * 0.0001, pm, pv * 0.0001), m, v
#
# class NeuBPR(nn.Module):
# 	def __init__(self, args, weight_decay=0.00001):
# 		super().__init__()
# 		self.name = args.model_name
# 		self.args = args
# 		# self.layers = [int(l) for l in args.layers.split('|')]
# 		self.layers = args.layers
#
# 		self.W_mlp = torch.nn.Embedding(num_embeddings=args.user_size, embedding_dim=args.user_emb_dim)
# 		self.H_mlp = torch.nn.Embedding(num_embeddings=args.item_size, embedding_dim=args.item_emb_dim)
# 		self.W_mf = torch.nn.Embedding(num_embeddings=args.user_size, embedding_dim=args.user_emb_dim)
# 		self.H_mf = torch.nn.Embedding(num_embeddings=args.item_size, embedding_dim=args.item_emb_dim)
#
# 		nn.init.xavier_normal_(self.W_mlp.weight.data)
# 		nn.init.xavier_normal_(self.H_mlp.weight.data)
# 		nn.init.xavier_normal_(self.W_mf.weight.data)
# 		nn.init.xavier_normal_(self.H_mf.weight.data)
#
# 		if self.name == 'NeuBPR':
# 			self.fc_layers = torch.nn.ModuleList()
# 			for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
# 				self.fc_layers.append(torch.nn.Linear(in_size, out_size))
#
# 			self.affine_output = torch.nn.Linear(in_features=self.layers[-1] + args.user_emb_dim, out_features=1)
#
# 		elif self.name == 'gmfBPR':
# 			self.affine_output = torch.nn.Linear(in_features=args.user_emb_dim, out_features=1)
#
# 		elif self.name == 'mlpBPR':
# 			self.fc_layers = torch.nn.ModuleList()
# 			for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
# 				self.fc_layers.append(torch.nn.Linear(in_size, out_size))
#
# 			self.affine_output = torch.nn.Linear(in_features=self.layers[-1], out_features=1)
#
# 		self.logistic = torch.nn.Sigmoid()
# 		self.weight_decay = args.weight_decay
#
# 	def forward(self, u, i, j):
# 		"""Return loss value.
#
#         Args:
#             u(torch.LongTensor): tensor stored user indexes. [batch_size,]
#             i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
#             j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
#
#         Returns:
#             torch.FloatTensor
#         """
# 		# print(u.size(), i.size())
#
# 		u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.size()[0], 1)
# 		i = torch.tensor(i, dtype=torch.int64).to(device).reshape(i.size()[0], 1)
# 		j = torch.tensor(j, dtype=torch.int64).to(device).reshape(j.size()[0], 1)
#
# 		x_ui = self.predict(u, i, mode='train')
# 		x_uj = self.predict(u, j, mode='train')
# 		x_uij = x_ui - x_uj
# 		# -------------------------------Mengyue Yang---------------------------------
# 		# # log_prob = F.logsigmoid(x_uij).mean()
# 		# log_prob = F.logsigmoid(x_uij)
#
# 		Wu_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
# 		Wu_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
#
# 		Hi_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
# 		Hi_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)
#
# 		Hj_mlp = self.H_mlp(j).reshape(j.size()[0], self.args.item_emb_dim)
# 		Hj_mf = self.H_mf(j).reshape(j.size()[0], self.args.item_emb_dim)
#
# 		# print('***'*10)
# 		# print('x_uij', x_uij.size(), Wu_mlp.size(), Hi_mlp.size())
# 		# print('***'*10)
#
# 		# log_prob = F.logsigmoid(x_uij).mean()
#
# 		# if self.args.model_name == 'NeuBPR':
# 		# 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
# 		# 		Wu_mf.norm(dim=1).pow(2).mean() + Hi_mlp.norm(dim=1).pow(2).mean() + \
# 		# 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean() + \
# 		# 		Hj_mf.norm(dim=1).pow(2).mean())
# 		# elif self.args.model_name in ['gmfBPR', 'bprBPR']:
# 		# 	regularization = self.weight_decay * (Wu_mf.norm(dim=1).pow(2).mean() + \
# 		# 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mf.norm(dim=1).pow(2).mean())
# 		# elif self.args.model_name == 'mlpBPR':
# 		# 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
# 		# 		Hi_mlp.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean())
#
# 		log_prob = F.logsigmoid(x_uij)
#
# 		if self.args.model_name == 'NeuBPR':
# 			regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Wu_mf.norm(dim=1) + Hi_mlp.norm(dim=1) + Hi_mf.norm(dim=1) + Hj_mlp.norm(dim=1) + Hj_mf.norm(dim=1))
# 		elif self.args.model_name in ['gmfBPR', 'bprBPR']:
# 			regularization = self.weight_decay * (Wu_mf.norm(dim=1) + Hi_mf.norm(dim=1) + Hj_mf.norm(dim=1))
# 		elif self.args.model_name == 'mlpBPR':
# 			regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Hi_mlp.norm(dim=1) + Hj_mlp.norm(dim=1))
# 		# ------------------------------------------------------------------------
# 		return -log_prob + regularization
#
# 	# ----------------------------Quanyu Dai----------------------------------
# 	def predict(self, u, i, mode='test'):
#
# 		if mode == 'test':
# 			u = torch.tensor(u, dtype=torch.int64).to(device_cpu).reshape(u.shape[0], 1)
# 			i = torch.tensor(i, dtype=torch.int64).to(device_cpu).reshape(i.shape[0], 1)
#
# 		if self.args.model_name == 'NeuBPR':
# 			user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
# 			item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
# 			user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
# 			item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)
#
# 			mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
# 			mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
#
# 			for idx, _ in enumerate(range(len(self.fc_layers))):
# 				mlp_vector = self.fc_layers[idx](mlp_vector)
# 				mlp_vector = torch.nn.ReLU()(mlp_vector)
#
# 			vector = torch.cat([mlp_vector, mf_vector], dim=-1)
#
# 		elif self.args.model_name == 'gmfBPR':
# 			# print(i)
# 			user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
# 			item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)
#
# 			vector = torch.mul(user_embedding_mf, item_embedding_mf)
#
# 		elif self.args.model_name == 'bprBPR':
# 			user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
# 			item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)
#
# 			vector = torch.mul(user_embedding_mf, item_embedding_mf)
#
# 		elif self.args.model_name == 'mlpBPR':
# 			user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
# 			item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
#
# 			vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
#
# 			for idx, _ in enumerate(range(len(self.fc_layers))):
# 				vector = self.fc_layers[idx](vector)
# 				vector = torch.nn.ReLU()(vector)
#
# 		# print('###'*10)
# 		# print('user_emb, item_emb, vector', user_embedding_mf.size(), item_embedding_mf.size(), vector.size())
# 		# print('###'*10)
#
# 		if self.args.model_name in ['NeuBPR', 'gmfBPR', 'mlpBPR']:
# 			logits = self.affine_output(vector)
# 			rating = logits.reshape(logits.size()[0])
# 		elif self.args.model_name == 'bprBPR':
# 			rating = vector.sum(dim=1)
# 			rating = rating.reshape(rating.size()[0])
#
# 		if mode == 'test':
# 			# rating = self.logistic(rating)
# 			rating = rating.detach().cpu().numpy()
#
# 		# print('rating', rating.shape, rating)
#
# 		return rating
#
# 	# ------------------------------------------------------------------------
#
# 	def load_pretrain_weights(self, gmf_model, mlp_model):
# 		"""Loading weights from trained MLP model & GMF model for NeuBPR"""
#
# 		self.W_mlp.weight.data = mlp_model.W_mlp.weight.data
# 		self.H_mlp.weight.data = mlp_model.H_mlp.weight.data
# 		for idx in range(len(self.fc_layers)):
# 			self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data
#
# 		self.W_mf.weight.data = gmf_model.W_mf.weight.data
# 		self.H_mf.weight.data = gmf_model.H_mf.weight.data
#
# 		self.affine_output.weight.data = 0.5 * torch.cat(
# 			[mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
# 		self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)