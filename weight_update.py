# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: weight_update.py
Author: liyang(gmlyytt@outlook.com)
Date: 2019/12/22 16:58:45
"""
import numpy as np
from collections import defaultdict


class WeightUpdate:
    CACHE = defaultdict()

    def __init__(self, weight_update_param=None):
        self.weight_update_func = None
        self._generate_weight_update_func(weight_update_param)

    def _generate_weight_update_func(self, weight_update_param=None):
        if weight_update_param is None:
            weight_update_param = {
                "mode": "momentum",
                "alpha": 0.01
            }
        self.weight_update_param = weight_update_param
        if weight_update_param["mode"] == "sgd":
            self.weight_update_func = self._sgd
        elif weight_update_param["mode"] == "momentum":
            self.weight_update_func = self._momentum
        else:
            pass

    @staticmethod
    def _sgd(W, delta_W, learning_rate, name=None):
        """SGD weight update.

        :param W:
        :param delta_W:
        :param learning_rate:
        :return:
        """
        if W.shape != delta_W.shape:
            return None

        W -= learning_rate * delta_W

        return W

    def _momentum(self, W, delta_W, learning_rate, name=None):
        """Momentum weight update.

        :param W:
        :param delta_W:
        :param learning_rate:
        :return:
        """
        if W.shape != delta_W.shape:
            return None

        if "alpha" not in self.weight_update_param:
            return None

        v = WeightUpdate.CACHE.get(name, np.zeros_like(W))

        alpha = self.weight_update_param["alpha"]
        WeightUpdate.CACHE[name] = -learning_rate * delta_W + alpha * v
        W += WeightUpdate.CACHE[name]

        return W
