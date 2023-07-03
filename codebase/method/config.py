from __future__ import print_function
import argparse
import time


def str2bool(v):
    return v is True or v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


parser.add_argument('--name', type=str, default='Debias', help='The main model name')
parser.add_argument('--time', type=str, default='', help='Current time')
# Learning process
parser.add_argument('--epoch_max', type=int, default=300, help='The learning epochs')
parser.add_argument('--intervention_epoch', type=int, default=5, help='The learning epochs of adversarial attack')

parser.add_argument('--iter_save', type=int, default=1, help='The save turn')
parser.add_argument('--pt_load_path', type=str, default='')

# Pretrain network
parser.add_argument('--train_mode', type=str, default='train', choices=['pretrain', 'train', 'test', 'robust'],
                    help='Weighted learning')
parser.add_argument('--use_weight', type=str2bool, default=False, help='Weighted learning')

# train network
parser.add_argument('--pretrain', type=bool, default=True, help="Pretrain SCM first")
parser.add_argument('--feature_data', type=bool, default=True, help="Side information")
parser.add_argument('--model_dir', type=str, default='', help='The model dir')
parser.add_argument('--user_feature_path', type=str, default=None, help="User feature path")
parser.add_argument('--item_feature_path', type=str, default=None, help="Item feature path")
# parser.add_argument('--downstream_model', type=str, default='base', choices=['base', 'MLP', 'DCN', 'MF'], help='The downstream model')
parser.add_argument('--upstream_model', type=str, default='base', choices=['base', 'MLP', 'DCN', 'bprBPR', 'mlpBPR', 'NeuBPR', 'MF'], help='The upstream model')
parser.add_argument('--cross_depth', type=int, default=2, help='The depth of cross layer')

# downstream is not MLP
parser.add_argument('--user_dim', type=int, default=300, help="User embedding dimension (equals to x_dim)")
parser.add_argument('--item_dim', type=int, default=211, help="Item embedding dimension")
parser.add_argument('--user_ori_dim', type=int, default=1, help="User embedding dimension")
parser.add_argument('--item_ori_dim', type=int, default=1, help="Item embedding dimension")
parser.add_argument('--user_size', type=int, default=15400, help="User number")
parser.add_argument('--prediction_size', type=int, default=2, help="Item number")
parser.add_argument('--user_item_size', type=int, nargs='+', default=[15400, 1000], help="User and Item size")
parser.add_argument('--user_emb_dim', type=int, default=32, help="Item feature dimension")
parser.add_argument('--item_emb_dim', type=int, default=32, help="Item feature dimension")


# Used to be an option, but now is solved
parser.add_argument('--x_dim', type=int, default=29, help="Feature embedding dimension")
parser.add_argument('--embedding_dim', type=int, default=64, help="Feature embedding dimension")
parser.add_argument('--enc_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                    help="Hidden layer dimension of encoder")
parser.add_argument('--dec_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                    help="Hidden layer dimension of decoder")
parser.add_argument('--downstream', type=str, default='MLP',
                    choices=['MLP', 'gmfBPR', 'bprBPR', 'mlpBPR', 'NeuBPR', 'LightGCN', 'DCN'], help="The mode of weight")
parser.add_argument('--random_start', type=bool, default=True, help="a random start for PDG")
parser.add_argument('--norm', type=str, default='2', choices=['2', 'inf'], help='Normalization')
parser.add_argument('--prior_type', type=str, default='conditional', choices=['conditional', 'standard', 'otherformat'],help='The prior of z')
parser.add_argument('--adversarial_type', type=str, default='normal', choices=['normal', 'adversarial'],help='The prior of z')


# data
parser.add_argument('--dataset', type=str, default='huawei')

parser.add_argument('--batch_size', type=int, default=8192)
parser.add_argument('--beta', type=float, default=1,
                    help='The weight of KL term')
parser.add_argument('--alpha', type=float, default=0.001,
                    help='Trade-off between out ball sample and others')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='Epsilon of the ball')
parser.add_argument('--rad', type=float, default=0.01,
                    help='Rad of random perturbation for PDG')
parser.add_argument('--bias', type=float, default=1,
                    help='The distance from 0')
parser.add_argument('--iv_lr', type=float, default=1e-4,
                    help='The weight of KL term')
parser.add_argument('--intervene_step', type=int, default=5,
                    help='Step to find worst case in wasserstein ball')
parser.add_argument('--mode', type=str, default='CausalRep',
                    choices=['CausalRep'], help="The mode of weight")
parser.add_argument('--attacker', type=str, default='pgd',
                    choices=['random', 'pgd'], help="The mode of weight")
parser.add_argument('--class_weight', type=int, nargs='+', default=[1, 10],
                    help="Class weight of soft cross entropy loss")


parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay in BPR model")
parser.add_argument('--dropout', type=float, default=0)
def get_config():
    config, unparsed = parser.parse_known_args()
    current_time = time.localtime(time.time())
    config.time = '{}_{}_{}_{}'.format(current_time.tm_mon, current_time.tm_mday, current_time.tm_hour,
                                       current_time.tm_min)
    model_name = [
        ('name={:s}', config.name),
        ('dataset={:s}', config.dataset),
    ]
    config.model_dir = '_'.join([t.format(v) for (t, v) in model_name])
    print('Loaded ./config.py')
    return config, unparsed


if __name__ == '__main__':
    # for debug of config
    config, unparsed = get_config()

'''
# how to use
from causal_controller.config import get_config as get_cc_config
cc_config,_=get_cc_config()
'''
