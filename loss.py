# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: loss.py
Author: liyang(gmlyytt@outlook.com)
Date: 2019/12/14 09:40:50
"""
import numpy as np


def square_loss(predictions, labels):
    data_size = predictions.shape[0]
    return 1.0 / data_size * (np.sum(labels - predictions) ** 2)


def square_loss_derivative(predictions, labels):
    data_size = predictions.shape[0]
    return 2.0 / data_size * (predictions - labels)
