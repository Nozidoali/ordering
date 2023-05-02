#!/usr/bin/env python
# -*- encoding=utf8 -*-

'''
Author: Hanyu Wang
Created time: 2023-05-01 12:16:55
Last Modified by: Hanyu Wang
Last Modified time: 2023-05-02 02:06:22
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

class DrawParams:
    omit_title: bool = True
    resolution: int = 10

colors = ['red', 'blue']
center_factor = 0.6

def plot_entity(ax, x: int, color):
    x = x + (1-center_factor) / 2 + center_factor * random.random()
    y = (1-center_factor) / 2 + center_factor * random.random()
    ax.scatter(x, y, color=color, s=20*DrawParams.resolution*DrawParams.resolution, marker='o')


def plot_frame(ax, num_cells: int, distribution: np.ndarray, potential: np.ndarray = None):
    plot_grid(ax, num_cells)
    if potential is not None:
        plot_potential(ax, num_cells, potential)
    plot_distribution(ax, num_cells, distribution)
    
def plot_distribution(ax, num_cells, distribution):
    for i in range(len(distribution)):
        color = colors[i % len(colors)]
        for j in range(num_cells):
            for _ in range(distribution[i][j]):
                plot_entity(ax, j, color)

def plot_grid(ax, num_cells: int):

    # Hide the tick labels and tick marks of the x and y axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set the limits
    ax.set_xlim(0, num_cells)
    ax.set_ylim(0, 1)

    # plot the grid
    for x in np.linspace(0, num_cells, num_cells+1):
        ax.plot([x, x], [0, 1], color='black', linewidth=DrawParams.resolution)

    ax.plot([0, num_cells], [0, 0], color='black', linewidth=DrawParams.resolution)
    ax.plot([0, num_cells], [1, 1], color='black', linewidth=DrawParams.resolution)

def plot_potential(ax, num_cells: int, potential: np.ndarray):

    max_potential = max(potential[0, :])
    min_potential = min(potential[0, :])

    for cell in range(num_cells):
        height = (potential[0,cell] - min_potential) / (max_potential - min_potential)

        height = 1 - height # flip the height

        alpha = height * 0.4 + 0.2
        
        rect = matplotlib.patches.Rectangle((cell,0), 1, height, color='red', alpha=alpha)
        
        ax.add_patch(rect)

if __name__ == '__main__':

    num_cells = 4

    fig, ax = plt.subplots(figsize=(num_cells, 1))
    
    plot_frame(ax, num_cells, np.array([[10,0,0,0],[0,0,0,10]]))
    plt.savefig('test.png')