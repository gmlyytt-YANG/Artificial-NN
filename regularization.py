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


def l2_derivative(input):
    return 2 * input


def l1_derivative(input):
    mask_pos = (input >= 0) * 1.0
    mask_neg = (input < 0) * -1.0

    return mask_pos + mask_neg


def regularization(input, regularization_option="l2"):
    if regularization_option == "l2":
        output = l2_derivative(input)
    elif regularization_option == "l1":
        output = l1_derivative(input)
    else:
        output = 0.0

    return output
