# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: data.py
Author: liyang(gmlyytt@outlook.com)
Date: 2019/12/14 10:17:50
"""
from sklearn.model_selection import train_test_split
import numpy as np

TRAIN_TEST_SPLIT_RATIO = 0.3


def load_data(input_dim, output_dim, num_of_data):
    X = np.random.randint(-10, 11, [num_of_data, input_dim])
    y = np.random.randint(0, 2, [num_of_data, output_dim])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=20191214)

    return X_train, X_test, y_train, y_test
