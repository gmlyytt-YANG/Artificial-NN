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
import numpy as np
from model import NnModel

# training params
LEARNING_RATE = 0.1
NUM_OF_ITERATIONS = 200

# data params
INPUT_DIM = 2
OUTPUT_DIM = 1
NUM_OF_DATA = 50


def run_app():
    nn = NnModel(LEARNING_RATE, INPUT_DIM, OUTPUT_DIM)
    X = np.random.rand(NUM_OF_DATA, INPUT_DIM)
    y = np.random.randint(0, 2, [NUM_OF_DATA, OUTPUT_DIM])

    nn.fit(X, y, NUM_OF_ITERATIONS)


if __name__ == '__main__':
    run_app()
