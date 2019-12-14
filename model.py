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

    def _forward(self, X, train=True):
        """Forward function of nn model, which computes the output of the nn model.

        :param X: array of shape [-1, self._input_dim]
        :param train: mode of forward, False means it is in the mode of validation.
        :return: output_layer, the result of forwarding.
        """
        if X.shape[1] != self._input_dim:
            print("_forward: input array dimension error")
            return None

        hidden_layer_before_activation = np.dot(X, self._hidden_layer_w)
        hidden_layer = self._activation_func_list["forward"](hidden_layer_before_activation)
        output_layer_before_activation = np.dot(hidden_layer, self._output_layer_w)
        output_layer = self._activation_func_list["forward"](output_layer_before_activation)

        if train:
            self._hidden_layer_before_activation = hidden_layer_before_activation
            self._hidden_layer = hidden_layer
            self._output_layer_before_activation = output_layer_before_activation
            self._output_layer = output_layer

        return output_layer

    def _compute_loss(self, output, y):
        """Compute loss

        :param output: the output of nn model.
        :param y: labels
        :return: True if there is no error, else False.
        """
        if output.shape != y.shape:
            print("_compute_loss: output shape is not y shape")
            return None

        loss = self._loss_func_list["forward"](output, y)

        return loss

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

    def evaluation(self, X, y):
        output = self._forward(X, train=False)
        if output is None:
            print("evaluation: _forward is wrong")
            return None

        loss = self._compute_loss(output, y)
        if loss is None:
            print("evaluation: _compute_loss is wrong")
            return None

        return loss

    def fit(self, X_train, y_train, X_valid, y_valid, num_of_iterations, verbose=False):
        """Fit the model according to input X and label y.

        :param X_train: input train data.
        :param X_valid: input valid data.
        :param y_train: train labels.
        :param y_valid: valid labels.
        :param num_of_iterations: self-define num of iterations.
        :param verbose: whether to show the training process.
        :return: True if there is no error, else False.
        """
        if X_train.shape[1] != self._input_dim or y_train.shape[1] != self._output_dim \
                or X_valid.shape[1] != self._input_dim or y_valid.shape[1] != self._output_dim:
            print("fit: input/output array dimension error")
            return False

        train_loss_list = []
        valid_loss_list = []

        valid_loss = self.evaluation(X_valid, y_valid)
        print("fit: validation loss at begin: {}".format(valid_loss))

        for _ in range(num_of_iterations):
            if self._forward(X_train) is None:
                print("fit: _forward wrong")
                return False

            if self._compute_loss(self._output_layer, y_train) is None:
                print("fit: _comput_loss wrong")
                return False

            if not self._backward(X_train, y_train):
                print("fit: _backward wrong")
                return False

            if _ % 50 == 0 or _ == num_of_iterations - 1:
                train_loss = self._compute_loss(self._output_layer, y_train)
                if train_loss is None:
                    print("fit: _compute_loss is wrong")
                    continue

                valid_loss = self.evaluation(X_valid, y_valid)
                if valid_loss is None:
                    print("fit: _compute_loss is wrong")
                    continue

                valid_loss_list.append(valid_loss)
                train_loss_list.append(train_loss)

                print("iteration {}, train loss: {}, valid_loss: {}".format(_, train_loss, valid_loss))

        if verbose:
            from matplotlib import pyplot as plt
            # plt.ylim(-10, 10)
            plt.plot(train_loss_list, label="train_loss")
            plt.plot(valid_loss_list, label="valid_loss")
            plt.legend()
            plt.show()
