# -*- coding:UTF-8 -*
#######################################
# training configs
#######################################
import argparse

import torch

parser = argparse.ArgumentParser(description='PyTorch CNN Sentence Classification')   
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='training optimizer (default: Adam)')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training (default: 100)')

parser.add_argument('--test-batch-size', type=int, default=32,
                    help='input batch size for testing (default: 100)')

parser.add_argument('--n-class', type=int, default=6,
                    help='number of class (default: 2)')

parser.add_argument('--age-class', type=int, default=3,
                    help='number of class (default: 2)')

parser.add_argument('--gender-class', type=int, default=2,
                    help='number of class (default: 2)')

parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train (default: 50)')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.001)')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')

parser.add_argument('--w-decay', type=float, default=0.0005,
                    help='L2 norm (default: 0)')

parser.add_argument('--log-interval', type=int, default=500,
                    help='how many batches to wait before logging training status')

parser.add_argument('--pre-trained', type=int, default=0,
                    help='using pre-trained model or not (default: 0)')

# data
parser.add_argument('--dataset', type=str, default='RAP_PETA',
                    help='current dataset')

# device
parser.add_argument('--cuda', type=int, default=1,
                    help='using CUDA training')

parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='using multi-gpu')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
params = "{}-{}-batch{}-epoch{}-lr{}-momentum{}-wdecay{}".format(args.dataset, args.optimizer, args.batch_size, args.epochs, args.lr, args.momentum, args.w_decay)
print('args: {}\nparams: {}'.format(args, params))

