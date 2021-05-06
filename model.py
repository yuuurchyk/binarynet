# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn

from module import BinaryLinear

class BinaryConnect(nn.Module):
    def __init__(self, in_features, out_features, num_units=2048):
        super(BinaryConnect, self).__init__()

        self.net = nn.Sequential(
                BinaryLinear(in_features, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15),
                nn.ReLU(),
                BinaryLinear(num_units, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15),
                nn.ReLU(),
                BinaryLinear(num_units, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15),
                nn.ReLU(),
                BinaryLinear(num_units, out_features),
                nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15),
                nn.LogSoftmax()
                )

    def forward(self, x):
        return self.net(x)
