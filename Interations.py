#!/usr/bin/env python
# -*- encoding=utf8 -*-

'''
Author: Hanyu Wang
Created time: 2023-05-01 13:03:38
Last Modified by: Hanyu Wang
Last Modified time: 2023-05-01 22:48:48
'''

import numpy as np

def get_trivial_interation(p, q):

    interations = np.zeros((2, 2), dtype=float)
    # initialize the interations
    interations[0][0] = p
    interations[0][1] = q
    interations[1][0] = q
    interations[1][1] = p

    return interations

def create_asymmetric_interaction(ratio):
    
    interations = np.zeros((2, 2), dtype=float)
    # initialize the interations
    interations[0][0] = -2 * ratio
    interations[0][1] = 2 * ratio
    interations[1][0] = -2 * ratio
    interations[1][1] = 1 * ratio

    return interations

def create_symmetric_interaction(p, q, ratio):
    
    interations = np.zeros((2, 2), dtype=float)
    # initialize the interations
    interations[0][0] = p * ratio
    interations[0][1] = q * ratio
    interations[1][0] = q * ratio
    interations[1][1] = p * ratio

    return interations

def create_example_interaction(p, ratio):
    
    q = (p-1)/2

    interations = np.zeros((2, 2), dtype=float)
    # initialize the interations
    interations[0][0] = p * ratio
    interations[0][1] = q * ratio
    interations[1][0] = q * ratio
    interations[1][1] = p * ratio

    return interations
