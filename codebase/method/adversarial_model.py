import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

logged_itter = 5


class PGD(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model#必须是pytorch的model
        self.device=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    def generate(self,x,**params):
        self.parse_params(**params)
        labels=self.y

        adv_x=self.attack(x,labels)
        return adv_x
    def parse_params(self,eps=0.3,iter_eps=0.01,nb_iter=40,clip_min=0.0,clip_max=1.0,C=0.0,
                     y=None,ord=np.inf,rand_init=True,flag_target=False):
        self.eps=eps
        self.iter_eps=iter_eps
        self.nb_iter=nb_iter
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.y=y
        self.ord=ord
        self.rand_init=rand_init
        self.model.to(self.device)
        self.flag_target=flag_target
        self.C=C


    def sigle_step_attack(self,x,pertubation,labels):
        adv_x=x+pertubation
        # get the gradient of x
        adv_x=Variable(adv_x)
        adv_x.requires_grad = True
        loss_func=nn.CrossEntropyLoss()
        preds=self.model(adv_x)
        if self.flag_target:
            loss =-loss_func(preds,labels)
        else:
            loss=loss_func(preds,labels)
            # label_mask=torch_one_hot(labels)
            #
            # correct_logit=torch.mean(torch.sum(label_mask * preds,dim=1))
            # wrong_logit = torch.mean(torch.max((1 - label_mask) * preds, dim=1)[0])
            # loss=-F.relu(correct_logit-wrong_logit+self.C)

        self.model.zero_grad()
        loss.backward()
        grad=adv_x.grad.data
        #get the pertubation of an iter_eps
        pertubation=self.iter_eps*np.sign(grad)
        adv_x=adv_x.cpu().detach().numpy()+pertubation.cpu().numpy()
        x=x.cpu().detach().numpy()

        pertubation=np.clip(adv_x,self.clip_min,self.clip_max)-x
        pertubation=clip_pertubation(pertubation,self.ord,self.eps)


        return pertubation
    def attack(self,x,labels):
        labels = labels.to(self.device)
        print(self.rand_init)
        if self.rand_init:
            x_tmp=x+torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        else:
            x_tmp=x
        pertubation=torch.zeros(x.shape).type_as(x).to(self.device)
        for i in range(self.nb_iter):
            pertubation=self.sigle_step_attack(x_tmp,pertubation=pertubation,labels=labels)
            pertubation=torch.Tensor(pertubation).type_as(x).to(self.device)
        adv_x=x+pertubation
        adv_x=adv_x.cpu().detach().numpy()

        adv_x=np.clip(adv_x,self.clip_min,self.clip_max)

        return adv_x


class InterventionVunerable(nn.Module):
    def __init__(self, args, decoder):
        super().__init__()
        self.name = 'IV'
        self.args = args

        self.decoder = decoder


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
        return loss  # torch.abs(neg_loss.detach() - loss)

class Attacker(nn.Module):
    def __init__(self, args, model, encoder=None):
        super().__init__()
        self.decoder = model
        self.args = args
        self.encoder = encoder

    def random_intervene(self, z, y=None):
        raw_loss = F.cross_entropy(self.decoder.predict(z), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        z_new = z.detach()
        z_neg = ut.random_perturb(z, epsilon=self.args.epsilon)
        neg_loss = F.cross_entropy(self.decoder.predict(z_neg), y, reduce=False).reshape(z.size()[0], 1).repeat(1,
                                                                                                                z.size()[
                                                                                                                    1])
        # replace = raw_loss < neg_loss
        # z_new[replace] = z_neg[replace]
        # print(z_neg[replace].size())
        z_new = torch.where(raw_loss < neg_loss, z_neg, z)
        return z_new

    def get_adv_examples(self, z, y, a=None):
        # Random start (to escape certain types of gradient masking)
        if self.args.random_start:
            z = self.random_intervene(z, y)
        # Keep track of the "best" (worst-case) loss and its
        # corresponding input
        best_loss = None
        best_z = None
        ori_input = z.clone().detach()
        # A function that updates the best loss and best input
        def replace_best(loss, bloss, z, bz):
            '''
            :param loss: loss of new z
            :param bloss: loss of exist worst z
            :param z: new z
            :param bz: exist worst z
            :return: exist z
            '''
            if bloss is None:
                bz = z.clone().detach()
                bloss = loss.clone().detach()
            else:
                # replace worst z
                bz = torch.where(loss > bloss, z.clone().detach(), bz.clone().detach())
                bloss = torch.where(loss > bloss, loss.clone().detach(), bloss.clone().detach())
            return bloss, bz

        # PGD iterates
        for _ in range(self.args.intervention_epoch):
            z = z.clone().detach().requires_grad_(True)
            losses = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
            assert losses.shape[0] == z.shape[0], \
                'Shape of losses must match input!'

            loss = torch.mean(losses)
            grad, = torch.autograd.grad(loss, [z])


            with torch.no_grad():
                args = [losses, best_loss, z, best_z]
                best_loss, best_z = replace_best(*args)  # if use_best else (losses, x)

                z = ut.step(z, grad)  # search new z by gradient method
                z = ut.project(z, ori_input, self.args.epsilon, norm=self.args.norm)  # normalization

        losses = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        args = [losses, best_loss, z, best_z]
        best_loss, best_z = replace_best(*args)
        return best_z

# old version  intervene x
# class Attacker(nn.Module):
#     def __init__(self, args, model, encoder=None):
#         super().__init__()
#         self.decoder = model
#         self.args = args
#         self.encoder = encoder
#
#     def random_intervene(self, x, y=None):
#         # print('hahahhahahahh')
#         if self.args.mode == 'CausalRep':
#             if self.args.prior_type == 'conditional':
#                 kl, m, v = self.encoder.kl(x, y)
#             else:
#                 kl, m, v = self.encoder.kl(x)
#             # v = torch.ones_like(m)*0.0001
#             z = ut.sample_gaussian(m, v)
#         else:
#             z = x
#         raw_loss = F.cross_entropy(self.decoder.predict(z), y, reduce=False).reshape(z.size()[0], 1).repeat(1, x.size()[1])
#         x_neg = ut.random_perturb(x, epsilon=self.args.epsilon)
#         if self.args.mode == 'CausalRep':
#             if self.args.prior_type == 'conditional':
#                 kl, m, v = self.encoder.kl(x_neg, y)
#             else:
#                 kl, m, v = self.encoder.kl(x_neg)
#             # v = torch.ones_like(m)*0.0001
#             z_neg = ut.sample_gaussian(m, v)
#         else:
#             z_neg = x
#
#         neg_loss = F.cross_entropy(self.decoder.predict(z_neg), y, reduce=False).reshape(z.size()[0], 1).repeat(1, x.size()[1])
#         # print(x.size(), x_neg.size())
#         x_new = torch.where(raw_loss < neg_loss, x_neg, x)
#         return x_new
#
#     def get_adv_examples(self, x, y, a=None):
#         if self.args.random_start:
#             x = self.random_intervene(x, y)
#
#         # Random start (to escape certain types of gradient masking)
#
#         # Keep track of the "best" (worst-case) loss and its
#         # corresponding input
#         best_loss = None
#         best_x = None
#         ori_input = x.clone().detach()
#         # A function that updates the best loss and best input
#         def replace_best(loss, bloss, x, bx):
#             '''
#             :param loss: loss of new x
#             :param bloss: loss of exist worst x
#             :param x: new x
#             :param bx: exist worst x
#             :return: exist x
#             '''
#             if bloss is None:
#                 bx = x.clone().detach()
#                 bloss = loss.clone().detach()
#             else:
#                 # replace worst x
#                 bx = torch.where(loss > bloss, x.clone().detach(), bx.clone().detach())
#                 bloss = torch.where(loss > bloss, loss.clone().detach(), bloss.clone().detach())
#             return bloss, bx
#
#
#         # PGD iterates
#         for _ in range(self.args.intervention_epoch):
#             #
#             # x = x.clone().detach().
#             if self.args.mode == 'CausalRep':
#                 x = x.clone().detach().requires_grad_(True)
#                 # print(self.encoder.predict(x))
#                 losses = F.cross_entropy(self.decoder.predict(self.encoder.predict(x)[0]), y, reduce=False).reshape(x.size()[0], 1).repeat(1, x.size()[1])
#                 #     m, v =
#                 # # v = torch.ones_like(m)*0.0001
#                 # m = m.clone().detach().requires_grad_(True)
#                 # v = v.clone().detach().requires_grad_(True)
#                 # z = ut.sample_gaussian(m, v)
#                 # pr
#             else:
#                 # z = x
#                 x = x.clone().detach().requires_grad_(True)
#                 losses = F.cross_entropy(self.decoder.predict(x), y, reduce=False).reshape(x.size()[0], 1).repeat(1, x.size()[1])
#             # print(self.decoder.predict(z, a), losses)
#             assert losses.shape[0] == x.shape[0], \
#                 'Shape of losses must match input!'
#
#             loss = torch.mean(losses)
#             # print(loss)
#
#             grad, = torch.autograd.grad(loss, [x], allow_unused=True)
#             # print(grad)
#
#
#             with torch.no_grad():
#                 args = [losses, best_loss, x, best_x]
#                 best_loss, best_x = replace_best(*args)  # if use_best else (losses, x)
#
#                 x = ut.step(x, grad)  # search new z by gradient method
#                 x = ut.project(x, ori_input, self.args.epsilon, norm=self.args.norm)  # normalization
#         if self.args.mode == 'CausalRep':
#             losses = F.cross_entropy(self.decoder.predict(self.encoder.predict(x)[0]), y,
#                                  reduce=False).reshape(x.size()[0], 1).repeat(1, x.size()[1])
#         else:
#             losses = F.cross_entropy(self.decoder.predict(x), y, reduce=False).reshape(x.size()[0], 1).repeat(1,
#                                                                                                               x.size()[
#                                                                                                                   1])
#         # losses = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(x.size()[0], 1).repeat(1, x.size()[1])
#         args = [losses, best_loss, x, best_x]
#         best_loss, best_x = replace_best(*args)
#         return best_x
class BaseRep(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.name = 'Rep'
        self.args = args

        self.encoder = encoder
        self.decoder = decoder

    def random_intervene(self, z, y=None):
        raw_loss = F.cross_entropy(self.decoder.predict(z), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        z_new = z.detach()
        z_neg = ut.random_perturb(z, epsilon=self.args.epsilon)

        neg_loss = F.cross_entropy(self.decoder.predict(z_neg), y, reduce=False).reshape(z.size()[0], 1).repeat(1,
                                                                                                                z.size()[
                                                                                                                    1])
        # replace = raw_loss < neg_loss
        # z_new[replace] = z_neg[replace]
        # print(z_neg[replace].size())
        z_new = torch.where(raw_loss < neg_loss, z_neg, z)
        #
        return z_new

    def get_adv_examples(self, z, y, a=None):
        # Random start (to escape certain types of gradient masking)
        if self.args.random_start:
            z = self.random_intervene(z, y)
        # Keep track of the "best" (worst-case) loss and its
        # corresponding input
        best_loss = None
        best_z = None
        ori_input = z.clone().detach()
        # A function that updates the best loss and best input
        def replace_best(loss, bloss, z, bz):
            '''
            :param loss: loss of new z
            :param bloss: loss of exist worst z
            :param z: new z
            :param bz: exist worst z
            :return: exist z
            '''
            if bloss is None:
                bz = z.clone().detach()
                bloss = loss.clone().detach()
            else:
                # replace worst z
                bz = torch.where(loss > bloss, z.clone().detach(), bz.clone().detach())
                bloss = torch.where(loss > bloss, loss.clone().detach(), bloss.clone().detach())
            return bloss, bz

        # PGD iterates
        for _ in range(self.args.intervention_epoch):
            z = z.clone().detach().requires_grad_(True)
            losses = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
            assert losses.shape[0] == z.shape[0], \
                'Shape of losses must match input!'

            loss = torch.mean(losses)
            grad, = torch.autograd.grad(loss, [z])


            with torch.no_grad():
                args = [losses, best_loss, z, best_z]
                best_loss, best_z = replace_best(*args)  # if use_best else (losses, x)

                z = ut.step(z, grad)  # search new z by gradient method
                z = ut.project(z, ori_input, self.args.epsilon, norm=self.args.norm)  # normalization

        losses = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        args = [losses, best_loss, z, best_z]
        best_loss, best_z = replace_best(*args)
        return best_z

    def inference_z(self, x, a=None):
        # forward
        if self.args.mode == 'IB':
            m, v = self.encoder.predict(x, a)
            z = ut.sample_gaussian(m, v)
        else:
            z = self.encoder.predict_normal(x, a)
        return z

    def generate_y(self, z, a=None):
        # forward
        y = self.decoder.predict(z, a)
        return y

    def neg_loss(self, x, y, a=None):
        # backward
        z = self.inference_z(x, y)

        ds_loss = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])

        return ds_loss.mean()


    def loss(self, x, y, a=None):
        # backward
        z = self.inference_z(x, y)

        z_neg = self.get_adv_examples(z, y, a)
        # print(z[0][0], z_neg[0][0])
        # assert z[0][0].data == z_neg[0][0].data

        ds_loss = self.decoder.loss(z_neg, y, a)

        z_rob_loss = torch.zeros_like(ds_loss)
        kl = torch.zeros_like(ds_loss)
        return ds_loss.mean(), ds_loss, kl, z, z_rob_loss


class InformationBottleneck(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.name = 'IB'
        self.args = args

        self.encoder = encoder
        self.decoder = decoder

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

    def outball(self, z, y=None, a=None):
        raw_loss = F.cross_entropy(self.decoder.predict(z), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        z_new = z.detach()
        z_neg = ut.random_perturb_out_ball(z, y, bias=self.args.bias, epsilon=self.args.epsilon)
        neg_loss = F.cross_entropy(self.decoder.predict(z_neg), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        z_new1 = z_new[torch.where(raw_loss > neg_loss)].reshape(-1, z.size()[1])
        y_new = y[torch.where(raw_loss[:, 0] > neg_loss[:, 0])]
        return z_new1, y_new

    def get_adv_examples(self, z, y, a=None):
        # Random start (to escape certain types of gradient masking)
        if self.args.random_start:
            z = self.random_intervene(z, y)
        # Keep track of the "best" (worst-case) loss and its
        # corresponding input
        best_loss = None
        best_z = None
        ori_input = z.clone().detach()
        # A function that updates the best loss and best input
        def replace_best(loss, bloss, z, bz):
            '''
            :param loss: loss of new z
            :param bloss: loss of exist worst z
            :param z: new z
            :param bz: exist worst z
            :return: exist z
            '''
            if bloss is None:
                bz = z.clone().detach()
                bloss = loss.clone().detach()
            else:
                # replace worst z
                bz = torch.where(loss > bloss, z.clone().detach(), bz.clone().detach())
                bloss = torch.where(loss > bloss, loss.clone().detach(), bloss.clone().detach())
            return bloss, bz

        # PGD iterates
        for _ in range(self.args.intervention_epoch):
            z = z.clone().detach().requires_grad_(True)
            losses = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
            assert losses.shape[0] == z.shape[0], \
                'Shape of losses must match input!'

            loss = torch.mean(losses)
            grad, = torch.autograd.grad(loss, [z])


            with torch.no_grad():
                args = [losses, best_loss, z, best_z]
                best_loss, best_z = replace_best(*args)  # if use_best else (losses, x)

                z = ut.step(z, grad)  # search new z by gradient method
                z = ut.project(z, ori_input, self.args.epsilon, norm=self.args.norm)  # normalization

        losses = F.cross_entropy(self.decoder.predict(z, a), y, reduce=False).reshape(z.size()[0], 1).repeat(1, z.size()[1])
        args = [losses, best_loss, z, best_z]
        best_loss, best_z = replace_best(*args)
        return best_z

    def inference_z(self, x, a=None):
        # forward
        m, v = self.encoder.predict(x, a)
        return m, v

    def generate_y(self, z, a=None):
        # forward
        y = self.decoder.predict(z, a)
        return y

    def neg_loss(self, x, y, a=None):
        if self.args.prior_type == 'conditional':
            kl, m, v = self.encoder.kl(x, y, a)
        else:
            kl, m, v = self.encoder.kl(x, a)
        # v = torch.ones_like(m)*0.0001
        z = ut.sample_gaussian(m, v)
        z_out, y = self.outball(z, y, a)
        # print(z_out.size(), y)
        if z_out.size()[0] == 0:
            return None
        # out_loss = -F.cross_entropy(self.decoder.predict(z_out), y)
        # print(z_out.size())
        out_loss = -F.cross_entropy(self.decoder.predict(z_out, a), y)

        return out_loss.mean()

    def loss(self, x, y, a=None):
        # backward
        if self.args.prior_type == 'conditional':
            kl, m, v = self.encoder.kl(x, y)
        else:
            kl, m, v = self.encoder.kl(x)
        # v = torch.ones_like(m)*0.0001
        z = ut.sample_gaussian(m, v)
        z_neg = self.get_adv_examples(z, y, a)
        # z_out = self.outball(z, y)

        ds_loss = self.decoder.loss(z_neg, y, a)
        # out_loss = -F.cross_entropy(self.decoder.predict(z_out), y)

        z_rob_loss = torch.zeros_like(ds_loss)
        return ds_loss.mean() + self.args.beta * kl.mean(), ds_loss, kl, z, z_rob_loss


class Encoder(nn.Module):  # q(z|x)
    def __init__(self, args):
        super().__init__()
        self.name = 'Encoder'
        self.args = args
        if self.args.mode in ['CausalRep', 'IB']:
            embedding_dim = self.args.embedding_dim * 2
        else:
            embedding_dim = self.args.embedding_dim
        if not self.args.feature_data:
            self.x_embedding_lookup = nn.Embedding(self.args.user_item_size[0], int(self.args.embedding_dim/2))
            self.a_embedding_lookup = nn.Embedding(self.args.user_item_size[1], int(self.args.embedding_dim/2))
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                in_dim = int(self.args.embedding_dim/2)
            else:
                in_dim = self.args.embedding_dim

            self.encode = nn.Sequential(
                nn.Linear(in_dim, self.args.enc_layer_dims[1]),
                nn.ELU(),
                nn.Linear(self.args.enc_layer_dims[1], embedding_dim)
            )
        else:
            self.encode = nn.Sequential(
                nn.Linear(self.args.x_dim, self.args.enc_layer_dims[0]),
                nn.ELU(),
                nn.Linear(self.args.enc_layer_dims[0], self.args.enc_layer_dims[1]),
                nn.ELU(),
                nn.Linear(self.args.enc_layer_dims[1], embedding_dim)
            )
    def embeddings(self, x):
        # print(x.size())
        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x[:, 0]).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x[:, 1]).reshape(-1, int(self.args.embedding_dim / 2))
        return torch.cat((x_emb, a_emb))

    def predict(self, x, a=None):
        # print(x.size())
        if self.args.upstream_model in ['bprBPR']:
            m = x
            v = torch.zeros_like(m)
        else:
            if not self.args.feature_data:
                # h = self.encode(self.embeddings(x))
                # print(x)
                h = self.encode(x)
            else:
                h = self.encode(x)

            m, v = ut.gaussian_parameters(h, dim=1)
            if self.args.mode == 'IBnorm':
                v = torch.ones_like(m)*0.001
            else:
                v = torch.zeros_like(m)
        return m, v

    def predict_normal(self, x, a=None):
        # print(x.size())
        if not self.args.feature_data:
            # h = self.encode(self.embeddings(x))
            # print(x)
            h = self.encode(x)
        else:
            h = self.encode(x)
        return h

    def kl(self, x, y=None, a=None):
        m, v = self.predict(x)

        if self.args.prior_type == 'conditional':
            pm, pv = ut.condition_prior([0, 1], y, m.size()[1])
        else:
            pm, pv = torch.zeros_like(m), torch.ones_like(m)
        # print(m)
        return ut.kl_normal(m, pv * 0.0001, pm, pv * 0.0001), m, v


class Decoder(nn.Module):  # p(y|z)
    def __init__(self, args):
        super().__init__()
        self.name = 'Decoder'
        self.args = args
        if self.args.upstream_model in ['bprBPR']:
            in_dim = int(self.args.embedding_dim/2)
        else:
            in_dim = self.args.embedding_dim
        self.decode = nn.Sequential(
            nn.Linear(in_dim, self.args.dec_layer_dims[1]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[1], self.args.dec_layer_dims[2]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[2], 2)
        )

        self.sigmd = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss()
        self.sftcross = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([self.args.class_weight[0], self.args.class_weight[1]])).float(),
            reduction='none')

    def predict(self, z, a=None):
        return self.decode(z)

    def loss(self, z, y, a=None):
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
            for idx, (in_size, out_size) in enumerate(zip(self.args.enc_layer_dims[:-1], self.args.enc_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.enc_layer_dims[-1] + args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'gmfBPR':
            self.affine_output = torch.nn.Linear(in_features=args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'mlpBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.args.enc_layer_dims[:-1], self.args.enc_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.enc_layer_dims[-1], out_features=1)

        self.logistic = torch.nn.Sigmoid()
        self.weight_decay = args.weight_decay
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.sftcross = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.args.class_weight[1]))

    def loss(self, u, y, i):
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

        if self.args.upstream_model == 'NeuBPR':
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

        elif self.args.upstream_model == 'gmfBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.upstream_model == 'bprBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.upstream_model == 'mlpBPR':
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

        if self.args.upstream_model in ['NeuBPR', 'gmfBPR', 'mlpBPR']:
            logits = self.affine_output(vector)
            rating = logits.reshape(logits.size()[0])
        elif self.args.upstream_model == 'bprBPR':
            rating = vector.sum(dim=1)
            rating = rating.reshape(rating.size()[0])

        if mode == 'test':
            # rating = self.logistic(rating)
            rating = rating#.detach().cpu().numpy()

        # print('rating', rating.shape, rating)

        return vector, torch.zeros_like(vector)

    def kl(self, x, y=None, a=None):
        m, v = self.predict(x)

        if self.args.prior_type == 'conditional':
            pm, pv = ut.condition_prior([0, 1], y, self.args.embedding_dim)
        else:
            pm, pv = torch.zeros_like(m), torch.ones_like(m)
        # print(m)
        return ut.kl_normal(m, pv * 0.0001, pm, pv * 0.0001), m, v


class CTRModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = 'Decoder'
        self.args = args
        if not self.args.feature_data:
            input_dim = self.args.embedding_dim
        else:
            input_dim = self.args.x_dim
        self.decode = nn.Sequential(
            nn.Linear(input_dim, self.args.dec_layer_dims[1]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[1], self.args.dec_layer_dims[2]),
            nn.ELU(),
            nn.Linear(self.args.dec_layer_dims[2], 2)
        )

        self.sigmd = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss()
        self.sftcross = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([self.args.class_weight[0], self.args.class_weight[1]])).float(),
            reduction='none')

    def predict(self, z, a=None):

        return self.decode(z)

    def loss(self, z, y, a=None):
        return self.sftcross(self.predict(z), y.squeeze_())


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
    def __init__(self, num_cont_fields, cross_depth, fc_dims=None,
                 dropout=None, batch_norm=None, out_type='binary', emb_dim=None, num_feats=None, num_cate_fields=None):
        super(DCN, self).__init__()
        # do not consider categorical in this version
        # self.emb_dim = emb_dim
        # self.num_feats = num_feats
        # self.num_cate_fields = num_cate_fields
        self.num_cont_fields = num_cont_fields

        self.cross_depth = cross_depth
        # embedding for category features
        # self.emb_layer = nn.Embedding(num_embeddings=num_feats - num_cont_fields, embedding_dim=emb_dim)
        # nn.init.xavier_uniform_(self.emb_layer.weight)

        # deep network
        if not fc_dims:
            fc_dims = [64, 32]
        self.fc_dims = fc_dims
        x0_dim = num_cont_fields  # + num_cate_fields * emb_dim
        self.deep = MLP(x0_dim, fc_dims, dropout, batch_norm)

        # cross network
        cross_layers = []
        for _ in range(cross_depth):
            cross_layers.append(CrossLayer(x0_dim))
        self.cross = nn.ModuleList(cross_layers)

        # self.out_layer = OutputLayer(in_dim=fc_dims[-1] + x0_dim, out_type=out_type)

    def embeddings(self, continuous_value, categorical_index=None):
        # cate_emb_value = self.emb_layer(categorical_index)  # N * num_cate_fields * emb_dim
        # # N * (num_cate_fields * emb_dim)
        # cate_emb_value = cate_emb_value.reshape((-1, self.num_cate_fields * self.emb_dim))
        x0 = continuous_value
        y_dnn = self.deep(x0)

        xi = x0
        for cross_layer in self.cross:
            xi = cross_layer(x0, xi)

        output = torch.cat([y_dnn, xi], dim=1)
        # output = self.out_layer(output)
        return output

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