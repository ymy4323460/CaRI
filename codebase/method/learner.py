

import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import Linear
import codebase.method.models as md
import codebase.method.adversarial_model as md

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


class Learner(nn.Module):
    def __init__(self, args, scm=None, repre_model=None, model=None):
        super().__init__()
        self.args = args
        self.name = self.args.name
        print('mode: {}-{}, upstream:{}, downstream:{}, dataset: {}, epsilon={}, bias={},  iv_lr={}'.format(
            args.mode, args.train_mode, args.upstream_model, args.downstream, args.dataset, args.epsilon, args.bias,
            args.iv_lr))
        if model is not None:
            self.model = model
        if self.args.mode == 'Noweight':
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.downstream == 'MLP':
                self.decoder = md.CTRModel(args).to(device)
            else:
                self.decoder = md.BPRclass(self.args).to(device)
        elif self.args.mode in ['CausalRep', 'IB']:

            if self.args.upstream_model == 'base':
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.upstream_model == 'DCN':
                self.encoder = md.DCN(self.args).to(device)
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                self.encoder = md.Encoder(self.args).to(device)

            if self.args.downstream == 'MLP':
                self.decoder = md.Decoder(self.args).to(device)
            else:
                self.decoder = md.BPRclass(self.args).to(device)
            self.ib = md.InformationBottleneck(self.args, self.encoder, self.decoder).to(device)
            self.iv = md.InterventionVunerable(self.args, self.decoder).to(device)
            if self.args.dataset == 'yahoo' and self.args.epsilon == 0.3:
                self.ib.args.epsilon = 0.5
        elif self.args.mode == 'Rep':
            if self.args.upstream_model == 'base':
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.upstream_model == 'DCN':
                self.encoder = md.DCN(self.args).to(device)
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.downstream == 'MLP':
                self.decoder = md.Decoder(self.args).to(device)
            else:
                self.decoder = md.BPRclass(self.args).to(device)
            self.ib = md.BaseRep(self.args, self.encoder, self.decoder).to(device)
        if args.train_mode == 'robust':
            if self.args.mode == 'Noweight':
                self.attacker = md.Attacker(args, self.decoder)
                self.test_attacker = md.Attacker(args, self.decoder)
            elif self.args.mode == 'CausalRep':
                self.attacker = md.Attacker(args, self.decoder, self.encoder)
                self.test_attacker = md.Attacker(args, self.decoder)
        if not self.args.feature_data:
            self.x_embedding_lookup = nn.Embedding(self.args.user_item_size[0], int(self.args.embedding_dim / 2)).to(
                device)
            self.a_embedding_lookup = nn.Embedding(self.args.user_item_size[1], int(self.args.embedding_dim / 2)).to(
                device)

    # def pretrain(self, x, y):
    #     if self.args.feature_data:
    #         x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0],self.args.x_dim])
    #     else:
    #         x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0],self.args.x_dim])
    #     y = torch.tensor(y, dtype=torch.float32).to(device)
    #     prediction_loss = self.model.loss(x, y)
    #     return residual

    def learn(self, x_raw, y, a=None, attack=False):
        if self.args.feature_data:
            x_raw = torch.tensor(x_raw, dtype=torch.float32).to(device).reshape([x_raw.size()[0], self.args.x_dim])
        else:
            x_raw = torch.tensor(x_raw, dtype=torch.int64).to(device).reshape([x_raw.size()[0], self.args.x_dim])
        y = torch.tensor(y, dtype=torch.int64).to(device)
        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x_raw[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x_raw[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', "NeuBPR"]:
                x_raw = torch.mul(x_emb, a_emb)
                weight_decay = self.args.weight_decay * (x_emb.norm(dim=1) + a_emb.norm(dim=1))
            else:
                x_raw = torch.cat((x_emb, a_emb), dim=1)

        # backward
        # print(x.size())
        if self.args.mode == 'Noweight':
            if self.args.train_mode == 'robust':

                x = self.attacker.get_adv_examples(x_raw, y)
                # print(x.size())
            else:
                x = x_raw
            return self.decoder.loss(x, y)
        elif self.args.mode in ['CausalRep', 'IB']:
            x = x_raw
            loss, ds_loss, kl, z, iv_loss = self.ib.loss(x, y)
            return loss.mean(), ds_loss.mean(), kl.mean(), z, iv_loss.mean()
        elif self.args.mode == 'Rep':
            x = x_raw
            # for name, param in self.ib.named_parameters():
            #     if name == 'encoder.encode.0.weight':
            #         # print(name, '      ', param[0][1])
            loss, ds_loss, kl, z, iv_loss = self.ib.loss(x, y)
            return loss, ds_loss.mean(), kl.mean(), z, iv_loss.mean()

    def learn_neg(self, x, y, a=None):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        y = torch.tensor(y, dtype=torch.int64).to(device)
        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                x = torch.mul(x_emb, a_emb)
            else:
                x = torch.cat((x_emb, a_emb), dim=1)
        # backward
        if self.args.mode in ['Noweight']:
            return self.decoder.loss(x, y)

        elif self.args.mode in ['CausalRep', 'Rep']:
            iv_loss = self.ib.neg_loss(x, y, a)
            if iv_loss is not None:
                # = self.decoder.loss(z, y)#self.iv.iv(z.detach(), y) #self.iv.iv(z, y)   torch.zeros_like(loss).mean()
                return iv_loss.mean()
            else:
                return None

    def predict(self, x, a=None):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        # forward
        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                x = torch.mul(x_emb, a_emb)
            else:
                x = torch.cat((x_emb, a_emb), dim=1)
        if self.args.mode == 'Noweight':
            y = self.decoder.predict(x)
            return (y, y) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y
        elif self.args.mode in ['CausalRep', 'IB']:
            m, v = self.ib.inference_z(x)
            # v = torch.ones_like(m)*0.0001
            z = ut.sample_gaussian(m, v)
            # print(v)
            y = self.ib.generate_y(z)
            return (y, z) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y
        elif self.args.mode == 'Rep':
            z = self.ib.inference_z(x)
            y = self.ib.generate_y(z)
            return (y, z) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y
        elif self.args.mode == 'Rep':
            z = self.ib.inference_z(x)
            y = self.ib.generate_y(z)
            return (y, z) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y

    def eval_adv(self, x, y, a=None):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        # forward
        y = torch.tensor(y, dtype=torch.int64).to(device)

        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                x = torch.mul(x_emb, a_emb)
            else:
                x = torch.cat((x_emb, a_emb), dim=1)
        if self.args.mode == 'Noweight':
            attacker = md.Attacker(self.args, self.decoder)
            attacker.args.epsilon = 0.3
            adv_x = attacker.get_adv_examples(x, y)
            y = self.decoder.predict(adv_x)
            return y
        elif self.args.mode in ['CausalRep', 'IB']:
            attacker = md.Attacker(self.args, self.decoder)
            attacker.args.epsilon = 0.3
            m, v = self.ib.inference_z(x)
            # v = torch.ones_like(m)*0.0001
            z = ut.sample_gaussian(m, v)
            adv_z = attacker.get_adv_examples(z, y)
            # print(v)
            y = self.ib.generate_y(adv_z)
            return y
        elif self.args.mode == 'Rep':
            attacker = md.Attacker(self.args, self.decoder)
            attacker.args.epsilon = 0.3
            z = self.ib.inference_z(x)
            adv_z = attacker.get_adv_examples(z, y)
            y = self.ib.generate_y(adv_z)
            return y
        # old version intervene x
        # if self.args.feature_data:
        #     x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        # else:
        #     x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        # y = torch.tensor(y, dtype=torch.int64).to(device)
        # # print(y, y.size())
        #
        # if self.args.mode == 'Noweight':
        #     attacker = md.Attacker(self.args, self.decoder)
        #     attacker.args.epsilon = 0.3
        # elif self.args.mode == 'CausalRep':
        #     attacker = md.Attacker(self.args, self.decoder, self.encoder)
        #     attacker.args.epsilon = 0.3
        # if not self.args.feature_data:
        #     x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
        #     a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
        #     x = torch.cat((x_emb, a_emb), dim=1)
        # adv_x = attacker.get_adv_examples(x, y)
        #
        # # print(adv_x)
        # if self.args.mode == 'Noweight':
        #     y = self.decoder.predict(adv_x)
        #     return y
        # else:
        #     m, v = self.ib.inference_z(adv_x)
        #     # v = torch.ones_like(m)*0.0001
        #     z = ut.sample_gaussian(m, v)
        #     # print(v)
        #     y = self.ib.generate_y(z)
        #     return y



import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import Linear
import codebase.method.models as md
import codebase.method.adversarial_model as md

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


class Learner(nn.Module):
    def __init__(self, args, scm=None, repre_model=None, model=None):
        super().__init__()
        self.args = args
        self.name = self.args.name
        print('mode: {}-{}, upstream:{}, downstream:{}, dataset: {}, epsilon={}, bias={},  iv_lr={}'.format(
            args.mode, args.train_mode, args.upstream_model, args.downstream, args.dataset, args.epsilon, args.bias,
            args.iv_lr))
        if model is not None:
            self.model = model
        if self.args.mode == 'Noweight':
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.downstream == 'MLP':
                self.decoder = md.CTRModel(args).to(device)
            else:
                self.decoder = md.BPRclass(self.args).to(device)
        elif self.args.mode in ['CausalRep', 'IB']:

            if self.args.upstream_model == 'base':
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.upstream_model == 'DCN':
                self.encoder = md.DCN(self.args).to(device)
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                self.encoder = md.Encoder(self.args).to(device)

            if self.args.downstream == 'MLP':
                self.decoder = md.Decoder(self.args).to(device)
            else:
                self.decoder = md.BPRclass(self.args).to(device)
            self.ib = md.InformationBottleneck(self.args, self.encoder, self.decoder).to(device)
            self.iv = md.InterventionVunerable(self.args, self.decoder).to(device)
            if self.args.dataset == 'yahoo' and self.args.epsilon == 0.3:
                self.ib.args.epsilon = 0.5
        elif self.args.mode == 'Rep':
            if self.args.upstream_model == 'base':
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.upstream_model == 'DCN':
                self.encoder = md.DCN(self.args).to(device)
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                self.encoder = md.Encoder(self.args).to(device)
            if self.args.downstream == 'MLP':
                self.decoder = md.Decoder(self.args).to(device)
            else:
                self.decoder = md.BPRclass(self.args).to(device)
            self.ib = md.BaseRep(self.args, self.encoder, self.decoder).to(device)
        if args.train_mode == 'robust':
            if self.args.mode == 'Noweight':
                self.attacker = md.Attacker(args, self.decoder)
                self.test_attacker = md.Attacker(args, self.decoder)
            elif self.args.mode == 'CausalRep':
                self.attacker = md.Attacker(args, self.decoder, self.encoder)
                self.test_attacker = md.Attacker(args, self.decoder)
        if not self.args.feature_data:
            self.x_embedding_lookup = nn.Embedding(self.args.user_item_size[0], int(self.args.embedding_dim / 2)).to(
                device)
            self.a_embedding_lookup = nn.Embedding(self.args.user_item_size[1], int(self.args.embedding_dim / 2)).to(
                device)

    # def pretrain(self, x, y):
    #     if self.args.feature_data:
    #         x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0],self.args.x_dim])
    #     else:
    #         x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0],self.args.x_dim])
    #     y = torch.tensor(y, dtype=torch.float32).to(device)
    #     prediction_loss = self.model.loss(x, y)
    #     return residual

    def learn(self, x_raw, y, a=None, attack=False):
        if self.args.feature_data:
            x_raw = torch.tensor(x_raw, dtype=torch.float32).to(device).reshape([x_raw.size()[0], self.args.x_dim])
        else:
            x_raw = torch.tensor(x_raw, dtype=torch.int64).to(device).reshape([x_raw.size()[0], self.args.x_dim])
        y = torch.tensor(y, dtype=torch.int64).to(device)
        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x_raw[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x_raw[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', "NeuBPR"]:
                x_raw = torch.mul(x_emb, a_emb)
                weight_decay = self.args.weight_decay * (x_emb.norm(dim=1) + a_emb.norm(dim=1))
            else:
                x_raw = torch.cat((x_emb, a_emb), dim=1)

        # backward
        # print(x.size())
        if self.args.mode == 'Noweight':
            if self.args.train_mode == 'robust':

                x = self.attacker.get_adv_examples(x_raw, y)
                # print(x.size())
            else:
                x = x_raw
            return self.decoder.loss(x, y)
        elif self.args.mode in ['CausalRep', 'IB']:
            x = x_raw
            loss, ds_loss, kl, z, iv_loss = self.ib.loss(x, y)
            return loss.mean(), ds_loss.mean(), kl.mean(), z, iv_loss.mean()
        elif self.args.mode == 'Rep':
            x = x_raw
            # for name, param in self.ib.named_parameters():
            #     if name == 'encoder.encode.0.weight':
            #         # print(name, '      ', param[0][1])
            loss, ds_loss, kl, z, iv_loss = self.ib.loss(x, y)
            return loss, ds_loss.mean(), kl.mean(), z, iv_loss.mean()

    def learn_neg(self, x, y, a=None):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        y = torch.tensor(y, dtype=torch.int64).to(device)
        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                x = torch.mul(x_emb, a_emb)
            else:
                x = torch.cat((x_emb, a_emb), dim=1)
        # backward
        if self.args.mode in ['Noweight']:
            return self.decoder.loss(x, y)

        elif self.args.mode in ['CausalRep', 'Rep']:
            iv_loss = self.ib.neg_loss(x, y, a)
            if iv_loss is not None:
                # = self.decoder.loss(z, y)#self.iv.iv(z.detach(), y) #self.iv.iv(z, y)   torch.zeros_like(loss).mean()
                return iv_loss.mean()
            else:
                return None

    def predict(self, x, a=None):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        # forward
        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                x = torch.mul(x_emb, a_emb)
            else:
                x = torch.cat((x_emb, a_emb), dim=1)
        if self.args.mode == 'Noweight':
            y = self.decoder.predict(x)
            return (y, y) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y
        elif self.args.mode in ['CausalRep', 'IB']:
            m, v = self.ib.inference_z(x)
            # v = torch.ones_like(m)*0.0001
            z = ut.sample_gaussian(m, v)
            # print(v)
            y = self.ib.generate_y(z)
            return (y, z) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y
        elif self.args.mode == 'Rep':
            z = self.ib.inference_z(x)
            y = self.ib.generate_y(z)
            return (y, z) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y
        elif self.args.mode == 'Rep':
            z = self.ib.inference_z(x)
            y = self.ib.generate_y(z)
            return (y, z) if self.args.dataset == 'celeba' or self.args.dataset[:3] == 'non' else y

    def eval_adv(self, x, y, a=None):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        # forward
        y = torch.tensor(y, dtype=torch.int64).to(device)

        if not self.args.feature_data:
            x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
            if self.args.upstream_model in ['bprBPR', 'NeuBPR']:
                x = torch.mul(x_emb, a_emb)
            else:
                x = torch.cat((x_emb, a_emb), dim=1)
        if self.args.mode == 'Noweight':
            attacker = md.Attacker(self.args, self.decoder)
            attacker.args.epsilon = 0.3
            adv_x = attacker.get_adv_examples(x, y)
            y = self.decoder.predict(adv_x)
            return y
        elif self.args.mode in ['CausalRep', 'IB']:
            attacker = md.Attacker(self.args, self.decoder)
            attacker.args.epsilon = 0.3
            m, v = self.ib.inference_z(x)
            # v = torch.ones_like(m)*0.0001
            z = ut.sample_gaussian(m, v)
            adv_z = attacker.get_adv_examples(z, y)
            # print(v)
            y = self.ib.generate_y(adv_z)
            return y
        elif self.args.mode == 'Rep':
            attacker = md.Attacker(self.args, self.decoder)
            attacker.args.epsilon = 0.3
            z = self.ib.inference_z(x)
            adv_z = attacker.get_adv_examples(z, y)
            y = self.ib.generate_y(adv_z)
            return y
        # old version intervene x
        # if self.args.feature_data:
        #     x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.x_dim])
        # else:
        #     x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], self.args.x_dim])
        # y = torch.tensor(y, dtype=torch.int64).to(device)
        # # print(y, y.size())
        #
        # if self.args.mode == 'Noweight':
        #     attacker = md.Attacker(self.args, self.decoder)
        #     attacker.args.epsilon = 0.3
        # elif self.args.mode == 'CausalRep':
        #     attacker = md.Attacker(self.args, self.decoder, self.encoder)
        #     attacker.args.epsilon = 0.3
        # if not self.args.feature_data:
        #     x_emb = self.x_embedding_lookup(x[:, 0].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
        #     a_emb = self.a_embedding_lookup(x[:, 1].to(device)).reshape(-1, int(self.args.embedding_dim / 2))
        #     x = torch.cat((x_emb, a_emb), dim=1)
        # adv_x = attacker.get_adv_examples(x, y)
        #
        # # print(adv_x)
        # if self.args.mode == 'Noweight':
        #     y = self.decoder.predict(adv_x)
        #     return y
        # else:
        #     m, v = self.ib.inference_z(adv_x)
        #     # v = torch.ones_like(m)*0.0001
        #     z = ut.sample_gaussian(m, v)
        #     # print(v)
        #     y = self.ib.generate_y(z)
        #     return y

