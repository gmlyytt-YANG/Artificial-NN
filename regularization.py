# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: regularization.py
Author: liyang(gmlyytt@outlook.com)
Date: 2019/12/14 10:17:50
"""


class Regularization:
    def __init__(self, regularization_param=None):
        self.regularization_func = None
        self._generate_regularization_func(regularization_param)

    def _generate_regularization_func(self, regularization_param=None):
        if regularization_param is None:
            regularization_param = {
                "alpha": 0.3,
                "mode": "l2"
            }
        self._alpha = regularization_param["alpha"]
        if regularization_param["mode"] == "l2":
            self.regularization_func = self._l2_derivative
        elif regularization_param["mode"] == "l1":
            self.regularization_func = self._l1_derivative
        else:
            pass

    def _l2_derivative(self, input):
        return 2 * self._alpha * input

    def _l1_derivative(self, input):
        mask_pos = (input >= 0) * 1.0
        mask_neg = (input < 0) * -1.0

        return self._alpha * (mask_pos + mask_neg)
