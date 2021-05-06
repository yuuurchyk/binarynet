# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn

from module import BinaryLinear

class BinaryConnect(nn.Module):
    def __init__(self, topology, batch_norm):
        super(BinaryConnect, self).__init__()

        assert len(topology) > 1
        for item in topology:
            assert item > 0

        layers = []
        for dim_in, dim_out in zip(topology[:-1], topology[1:]):
            layers.append(BinaryLinear(dim_in, dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out, eps=1e-4, momentum=0.15))
            layers.append(nn.ReLU())

        layers.pop()
        layers.append(nn.LogSoftmax())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
