# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import


import argparse
import train

parser = argparse.ArgumentParser(description='Deterministic Binary Connect on MNIST')
parser.add_argument('--cuda', default=False,
        action='store_true', help='Whether to use cuda')
parser.add_argument('--topology', nargs="+", type=int, help='Model topology', required=True)
parser.add_argument('--batch_norm', default=False, action='store_true',
        help='Whether to include batch norm layers into the network')
parser.add_argument('--batch_size', type=int, default=100,
        help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1000,
        help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
        help='Learning rate')
parser.add_argument('--epochs', type=int, default=20,
        help='Epochs')
args = parser.parse_args()

train.train(args)
