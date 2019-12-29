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
from collections import defaultdict

from activation import ActivationLoader
from loss import LossLoader
from regularization import Regularization
from weight_update import WeightUpdate


class NnModel:
    def __init__(self,
                 learning_rate,
                 input_dim,
                 output_dim,
                 layer_unit_num_list=None,
                 activation_method="sigmoid",
                 loss_method="square",
                 regularization_param=None,
                 weight_update_param=None):
        self._learning_rate = learning_rate if learning_rate is not None else 0
        self._input_dim = input_dim if input_dim is not None else 3
        self._output_dim = output_dim if output_dim is not None else 1

        if layer_unit_num_list is None:
            layer_unit_num_list = [5, 3]
        layer_unit_num_list.append(output_dim)
        self.layer_unit_num_list = layer_unit_num_list

        self.layer_num = len(self.layer_unit_num_list)

        self._layers = defaultdict()
        self._outputs = defaultdict()

        self._loss = None
        self._regularization_param = regularization_param
        self._weight_update_param = weight_update_param

        self._generate_network_structure()
        self._activation_func_list = ActivationLoader(activation_method).activation_func_list
        self._loss_func_list = LossLoader(loss_method).loss_func_list

    def _generate_network_structure(self):
        for layer_index in range(self.layer_num):
            input_dim = self._input_dim if layer_index == 0 else self.layer_unit_num_list[layer_index - 1]

            self._layers["weight_{}".format(layer_index)] = np.random.rand(input_dim,
                                                                           self.layer_unit_num_list[layer_index])
            # if self._weight_update_param["mode"] == "momentum":
            #     self._layers["weight_{}_v".format(layer_index)] = np.zeros([input_dim, self.layer_num[layer_index]])

    def _forward(self, X, train=True):
        """Forward function of nn model, which computes the output of the nn model.

        :param X: array of shape [-1, self._input_dim]
        :param train: mode of forward, False means it is in the mode of validation.
        :return: output_layer, the result of forwarding.
        """
        if X.shape[1] != self._input_dim:
            print("_forward: input array dimension error")
            return None, False

        outputs = defaultdict()
        for layer_index in range(self.layer_num):
            input_data = X if layer_index == 0 else outputs["output_{}".format(layer_index - 1)]
            outputs["output_before_activation_{}".format(layer_index)] = \
                np.dot(input_data, self._layers["weight_{}".format(layer_index)])
            outputs["output_{}".format(layer_index)] = \
                self._activation_func_list["forward"](outputs["output_before_activation_{}".format(layer_index)])

        if train:
            self._outputs = outputs
            return None, True

        return outputs["output_{}".format(self.layer_num - 1)], True

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
        if self._outputs["output_{}".format(self.layer_num - 1)].shape != y.shape:
            print("_backward_loss: self._output_layer.shape != y.shape")
            return False

        d_outputs = defaultdict()
        d_layers = defaultdict()

        for layer_index in range(self.layer_num - 1, -1, -1):
            # compute gradients
            if layer_index == self.layer_num - 1:
                d_outputs["output_{}".format(layer_index)] = self._loss_func_list["backward"](
                    self._outputs["output_{}".format(layer_index)], y)
            else:
                d_outputs["output_{}".format(layer_index)] = \
                    np.dot(d_outputs["output_before_activation_{}".format(layer_index + 1)],
                           self._layers["weight_{}".format(layer_index + 1)].T)

            d_outputs["output_before_activation_{}".format(layer_index)] = \
                self._activation_func_list["forward"](
                    self._outputs["output_before_activation_{}".format(layer_index)]) * d_outputs[
                    "output_{}".format(layer_index)]

            input_for_weight = X if layer_index == 0 else self._outputs["output_{}".format(layer_index - 1)]
            d_layers["weight_{}".format(layer_index)] = np.dot(input_for_weight.T, d_outputs[
                "output_before_activation_{}".format(layer_index)])

            # regularization
            d_layers["weight_{}".format(layer_index)] += \
                Regularization(self._regularization_param).regularization_func(
                    self._layers["weight_{}".format(layer_index)])

            # update parameters
            self._layers["weight_{}".format(layer_index)] = WeightUpdate(self._weight_update_param).weight_update_func(
                self._layers["weight_{}".format(layer_index)], d_layers["weight_{}".format(layer_index)],
                self._learning_rate, name="weight_{}".format(layer_index))

        return True

    def evaluation(self, X, y):
        output = self.predict(X)
        loss = self._compute_loss(output, y)
        if loss is None:
            print("evaluation: _compute_loss is wrong")
            return None

        return loss

    def predict(self, X):
        output, succ = self._forward(X, train=False)
        if not succ:
            print("evaluation: _forward is wrong")
            return None

        return output

    def fit(self, data_dict, num_of_iterations, early_stopping_iter=300, verbose=False):
        """Fit the model according to input X and label y.

        :param data_dict:
        :param num_of_iterations: self-define num of iterations.
        :param early_stopping_iter:
        :param verbose: whether to show the training process.
        :return: True if there is no error, else False.
        """
        X_train, y_train, X_valid, y_valid = \
            data_dict["X_train"], data_dict["y_train"], data_dict["X_evaluation"], data_dict["y_evaluation"]
        if X_train.shape[1] != self._input_dim or y_train.shape[1] != self._output_dim \
                or X_valid.shape[1] != self._input_dim or y_valid.shape[1] != self._output_dim:
            print("fit: input/output array dimension error")
            return False

        train_loss_list = []
        valid_loss_list = []

        valid_loss = self.evaluation(X_valid, y_valid)
        print("fit: validation loss at begin: {}".format(valid_loss))

        min_valid_loss = 100000000
        iter_tolerate = 0
        for index in range(num_of_iterations):
            _, succ = self._forward(X_train)
            if not succ:
                print("fit: _forward wrong")
                return False

            train_loss = self._compute_loss(self._outputs["output_{}".format(self.layer_num - 1)], y_train)
            if train_loss is None:
                print("fit: _comput_loss wrong")
                return False

            if not self._backward(X_train, y_train):
                print("fit: _backward wrong")
                return False

            valid_loss = self.evaluation(X_valid, y_valid)

            # early_stopping
            if valid_loss >= min_valid_loss:
                if iter_tolerate >= early_stopping_iter:
                    print("iteration {} fit: arrive limitation of early stopping".format(index))
                    break
                else:
                    iter_tolerate += 1
            else:
                min_valid_loss = valid_loss

            if index % 50 == 0 or index == num_of_iterations - 1:
                if valid_loss is None:
                    print("fit: _compute_loss is wrong")
                    continue

                valid_loss_list.append(valid_loss)
                train_loss_list.append(train_loss)

                print("iteration {}, train loss: {}, valid_loss: {}".format(index, train_loss, valid_loss))

        if verbose:
            from matplotlib import pyplot as plt
            # plt.ylim(-10, 10)
            plt.plot(train_loss_list, label="train_loss")
            plt.plot(valid_loss_list, label="valid_loss")
            plt.legend()
            plt.show()
