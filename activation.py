# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: activation.py
Author: liyang(gmlyytt@outlook.com)
Date: 2019/12/14 09:40:50
"""

import numpy as np


def sigmoid(X):
    """Sigmoid function.

    :param X: input.
    :return: sigmoid output.
    """
    return 1.0 / (1.0 + np.exp(-X))


def sigmoid_derivative(X):
    """Sigmoid derivative function.

    :param X: input.
    :return: sigmoid derivative output.
    """
    return sigmoid(X) * (1 - sigmoid(X))
