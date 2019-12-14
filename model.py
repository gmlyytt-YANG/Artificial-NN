# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: model.py
Author: liyang(gmlyytt@outlook.com)
Date: 2019/12/14 09:40:50
"""

import numpy as np

from activation import sigmoid, sigmoid_derivative
from loss import square_loss, square_loss_derivative


class NnModel:
    def __init__(self,
                 learning_rate,
                 input_dim,
                 output_dim,
                 hidden_layer_unit_num=3):
        self._learning_rate = learning_rate if learning_rate is not None else 0

        self._input_dim = input_dim if input_dim is not None else 3
        self._output_dim = output_dim if output_dim is not None else 1

        self._hidden_layer_w = np.random.rand(self._input_dim, hidden_layer_unit_num)
        self._hidden_layer_before_activation = None
        self._hidden_layer = None

        self._output_layer_w = np.random.rand(hidden_layer_unit_num, self._output_dim)
        self._output_layer_before_activation = None
        self._output_layer = None

        self._activation_func_list = {
            "forward": sigmoid,
            "backward": sigmoid_derivative
        }

        self._loss = None
        self._loss_func_list = {
            "forward": square_loss,
            "backward": square_loss_derivative
        }

    def _forward(self, X):
        """Forward function of nn model, which computes the output of the nn model.

        :param X: array of shape [-1, self._input_dim]
        :return: True if there is no error, else False.
        """
        if X.shape[1] != self._input_dim:
            print("_forward: input array dimension error")
            return False
        self._hidden_layer_before_activation = np.dot(X, self._hidden_layer_w)
        self._hidden_layer = self._activation_func_list["forward"](self._hidden_layer_before_activation)
        self._output_layer_before_activation = np.dot(self._hidden_layer, self._output_layer_w)
        self._output_layer = self._activation_func_list["forward"](self._output_layer_before_activation)

        return True

    def _compute_loss(self, output, y):
        """Compute loss

        :param output: the output of nn model.
        :return: True if there is no error, else False.
        """
        if output.shape != y.shape:
            print("_compute_loss: output shape is not y shape")
            return False
        self._loss = self._loss_func_list["forward"](output, y)

        return True

    def _backward(self, X, y):
        """Backward function of nn model

        :param X:
        :param y:
        :return: True if there is no error, else False.
        """
        if self._output_layer.shape != y.shape:
            print("_backward_loss: self._output_layer.shape != y.shape")
            return False
        d_loss_d_output_layer = self._loss_func_list["backward"](self._output_layer, y)

        d_loss_d_output_layer_before_activation = \
            self._activation_func_list["backward"](self._output_layer_before_activation) * d_loss_d_output_layer

        d_loss_d_output_layer_w = np.dot(self._hidden_layer.T, d_loss_d_output_layer_before_activation)
        d_loss_d_hidden_layer = np.dot(d_loss_d_output_layer_before_activation, self._output_layer_w.T)

        d_loss_d_hidden_layer_before_activation = \
            self._activation_func_list["backward"](self._hidden_layer_before_activation) * d_loss_d_hidden_layer

        d_loss_d_hidden_layer_w = np.dot(X.T, d_loss_d_hidden_layer_before_activation)

        self._output_layer_w -= self._learning_rate * d_loss_d_output_layer_w
        self._hidden_layer_w -= self._learning_rate * d_loss_d_hidden_layer_w

        return True

    def fit(self, X, y, num_of_iterations):
        """Fit the model according to input X and label y.

        :param X: input data.
        :param y: labels.
        :param num_of_iterations: self-define num of iterations.
        :return: True if there is no error, else False.
        """
        if X.shape[1] != self._input_dim or y.shape[1] != self._output_dim:
            print("fit: input/output array dimension error")
            return False

        for _ in range(num_of_iterations):
            if not self._forward(X):
                print("fit: _forward wrong")
                return False

            if not self._compute_loss(self._output_layer, y):
                print("fit: _comput_loss wrong")
                return False

            if not self._backward(X, y):
                print("fit: _backward wrong")
                return False

            print(self._loss)
