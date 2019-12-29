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


class LossLoader:
    def __init__(self, loss_source):
        self.loss_func_list = None
        self._generate_loss_func(loss_source)

    def _generate_loss_func(self, loss_source):
        if loss_source == "square":
            self.loss_func_list = {
                "forward": self.square_loss,
                "backward": self.square_loss_derivative
            }
        else:
            pass

    @staticmethod
    def square_loss(predictions, labels):
        data_size = predictions.shape[0]
        return 1.0 / data_size * (np.sum(labels - predictions) ** 2)

    @staticmethod
    def square_loss_derivative(predictions, labels):
        data_size = predictions.shape[0]
        return 2.0 / data_size * (predictions - labels)
