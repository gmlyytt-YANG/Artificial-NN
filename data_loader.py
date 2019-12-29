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
import pandas as pd

TRAIN_TEST_SPLIT_RATIO = 0.3
DATA_FILE = "./data/Seed_Data.csv"


class DataLoader:
    def __init__(self, data_source="random"):
        # class member definition
        self.train_evaluation_data = None
        self.train_evaluation_param = None

        # choose data source
        if data_source == "random":
            self._load_data_random()
        elif data_source == "seed_from_uci":
            self._load_data_seed_from_uci()
        else:
            pass

    def _generate_train_evaluation_data(self, X, y):
        X_train, X_evaluation, y_train, y_evaluation = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO,
                                                                        random_state=20191214)
        self.train_evaluation_data = {
            "X_train": X_train,
            "X_evaluation": X_evaluation,
            "y_train": y_train,
            "y_evaluation": y_evaluation
        }
        self.train_evaluation_param = {
            "input_dim": X.shape[-1],
            "output_dim": y.shape[-1]
        }

    def _load_data_random(self):
        """Generate random data.

        :return:
        """
        # data params
        input_dim = 500
        output_dim = 3
        num_of_data = 50000

        X = np.random.randint(-10, 11, [num_of_data, input_dim])
        y = np.random.randint(0, 2, [num_of_data, output_dim])

        self._generate_train_evaluation_data(X, y)

    # TODO: add softmax_entropy loss to evaluation this dataset
    def _load_data_seed_from_uci(self):
        """Generate data from https://www.kaggle.com/dongeorge/seed-from-uci

        :return:
        """
        data = pd.read_csv(DATA_FILE)
        target_col = ["target"]
        data_col = [_ for _ in data.columns if _ not in target_col]
        X = data[data_col].values
        y = data[target_col].values

        self._generate_train_evaluation_data(X, y)
