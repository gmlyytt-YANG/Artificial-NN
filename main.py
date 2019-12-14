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
from data import load_data

# training params
LEARNING_RATE = 0.1
NUM_OF_ITERATIONS = 2000

# data params
INPUT_DIM = 500
OUTPUT_DIM = 3
NUM_OF_DATA = 50000


def run_app():
    nn = NnModel(LEARNING_RATE, INPUT_DIM, OUTPUT_DIM)
    X_train, X_test, y_train, y_test = load_data(INPUT_DIM, OUTPUT_DIM, NUM_OF_DATA)

    nn.fit(X_train, y_train, X_test, y_test, NUM_OF_ITERATIONS, verbose=True)


if __name__ == '__main__':
    run_app()
