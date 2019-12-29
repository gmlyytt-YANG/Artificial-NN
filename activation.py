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


class ActivationLoader:
    def __init__(self, activation_source):
        self.activation_func_list = None
        self._generate_activation_func(activation_source)

    def _generate_activation_func(self, activation_source):
        if activation_source == "sigmoid":
            self.activation_func_list = {
                "forward": self.sigmoid,
                "backward": self.sigmoid_derivative
            }
        else:
            pass

    @staticmethod
    def sigmoid(X):
        """Sigmoid function.

        :param X: input.
        :return: sigmoid output.
        """
        return 1.0 / (1.0 + np.exp(-X))

    @staticmethod
    def sigmoid_derivative(X):
        """Sigmoid derivative function.

        :param X: input.
        :return: sigmoid derivative output.
        """
        return ActivationLoader.sigmoid(X) * (1 - ActivationLoader.sigmoid(X))
