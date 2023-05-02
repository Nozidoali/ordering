#!/usr/bin/env python
# -*- encoding=utf8 -*-

'''
Author: Hanyu Wang
Created time: 2023-05-01 15:09:36
Last Modified by: Hanyu Wang
Last Modified time: 2023-05-01 15:11:46
'''

import numpy as np


def create_uniform_distribution(num_cells: int, num_entities: int):
    # sub population distribution is a matrix with shape (num_subpopulation, num_cells)
    subpopulation_distribution = np.ones((2, num_cells), dtype=int) * num_entities // num_cells

    return subpopulation_distribution

def create_polar_distribution(num_cells: int, num_entities: int):
    # sub population distribution is a matrix with shape (num_subpopulation, num_cells)
    subpopulation_distribution = np.zeros((2, num_cells), dtype=int)

    # initialize the sub population distribution
    subpopulation_distribution[0][0] = num_entities
    subpopulation_distribution[1][-1] = num_entities

    return subpopulation_distribution