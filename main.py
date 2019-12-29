# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: main.py
Author: liyang(gmlyytt@outlook.com)
Date: 2019/12/14 10:17:50
"""
from model import NnModel
from data_loader import DataLoader

# training params
LEARNING_RATE = 0.01
NUM_OF_ITERATIONS = 20000


def run_app():
    data_loader = DataLoader("random")

    nn = NnModel(LEARNING_RATE,
                 data_loader.train_evaluation_param["input_dim"],
                 data_loader.train_evaluation_param["output_dim"],
                 layer_unit_num_list=[5, 3],
                 activation_method="sigmoid",
                 loss_method="square",
                 regularization_param={"alpha": 0.3, "mode": "l2"},
                 weight_update_param={"mode": "momentum", "alpha": 0.01})

    nn.fit(data_loader.train_evaluation_data,
           num_of_iterations=NUM_OF_ITERATIONS,
           verbose=True)


if __name__ == '__main__':
    run_app()
