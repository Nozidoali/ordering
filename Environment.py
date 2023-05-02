#!/usr/bin/env python
# -*- encoding=utf8 -*-

'''
Author: Hanyu Wang
Created time: 2023-05-01 12:04:37
Last Modified by: Hanyu Wang
Last Modified time: 2023-05-02 03:36:00
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import imageio

from DrawUtil import *
from Interations import *
from Distributions import *

import os

class Params:
    fluctuation: float = 0.00
    diffusion: float = 0.00
    fluctuation_rate: float = 1.0
    P: float = -0.5
    rate: float = 0.1
    
    num_iterations: int = 500
    frames_to_record: int = 100
    num_entities: int = 100

    num_cells: int = 6

    plot_potential: bool = True

    # distribution_initialization = 'polar'
    distribution_initialization = 'uniform'

    loop_gif: bool = 0

class Entity:
    def __init__(self, x, subpopulation_idx) -> None:
        self.x = x
        self.subpopulation_idx = subpopulation_idx

class Environment:

    def __init__(self) -> None:
        
        self.num_cells = Params.num_cells
        self.num_subpopulation = 2

        self.num_iterations = Params.num_iterations

        self.frames_to_record = Params.frames_to_record
        self.num_entities = Params.num_entities
        
        if self.num_subpopulation != 2:
            raise NotImplementedError("Currently, only support two subpopulations")

        # sub population distribution is a matrix with shape (num_subpopulation, num_cells)
        if Params.distribution_initialization == 'polar':
            self.subpopulation_distribution = create_polar_distribution(self.num_cells, self.num_entities)
        elif Params.distribution_initialization == 'uniform':
            self.subpopulation_distribution = create_uniform_distribution(self.num_cells, self.num_entities//self.num_subpopulation)
        else:
            raise NotImplementedError("Currently, only support polar and uniform distribution initialization")

        # potential
        self.potential = np.zeros((self.num_subpopulation, self.num_cells), dtype=float)

        # the interations
        # self.interations = create_asymmetric_interaction(Params.rate)
        # self.interations = create_symmetric_interaction(Params.P, Params.Q, Params.rate)
        self.interations = create_example_interaction(Params.P, Params.rate)

        # entities
        self.diffusion_direction = np.zeros((self.num_subpopulation), dtype=int)
        self.diffusion_direction[0] = 1
        self.diffusion_direction[1] = -1
        self.diffusion_velocity = np.zeros((self.num_subpopulation), dtype=float)
        self.diffusion_velocity[0] = Params.diffusion
        self.diffusion_velocity[1] = Params.diffusion

        # fluctuations
        self.flucturations = np.zeros((self.num_subpopulation), dtype=float)
        self.flucturations[0] = Params.fluctuation * Params.fluctuation_rate
        self.flucturations[1] = Params.fluctuation * Params.fluctuation_rate

        # 
        self.saved_iterations = []

    def save_iteration(self, iter: int):
        fig, ax = plt.subplots(figsize=(self.num_cells * DrawParams.resolution, 2 * DrawParams.resolution))  # Create a Figure and an Axes object
        plt.subplots_adjust(top=0.627,
            bottom=0.199,
            left=0.066,
            right=0.95,
            hspace=0.2,
            wspace=0.2)
        
        if not DrawParams.omit_title:
            ax.set_title(f'Iteration {iter:04d}')

        if Params.plot_potential:
            plot_frame(ax, self.num_cells, self.subpopulation_distribution, self.potential)
        else:
            plot_frame(ax, self.num_cells, self.subpopulation_distribution)
        plt.savefig(f'img/{iter}.png')
        plt.close(fig)
        self.saved_iterations.append(iter)

    def simulate(self):
        # remove all the saved images
        os.system('rm -rf ./img/*')
        
        iterval = self.num_iterations // self.frames_to_record
        self.update_potential()
        for i in range(self.num_iterations):
            if i % iterval == 0:
                print(f'Iteration {i:04d}\n', end='')
                print(self.potential)
                self.save_iteration(i)
            self.update()
            self.update_potential()
                
    def export_gif(self):
        frames = []
        for i in self.saved_iterations:
            image = imageio.imread(f'./img/{i}.png')
            frames.append(image)
        imageio.mimsave('./example.gif', # output gif
                frames,                  # array of input frames
                fps = 10,                # optional: frames per second
                loop = Params.loop_gif   # optional: loop gif
        )

    def update_potential(self):
        self.potential = np.matmul(self.interations, self.subpopulation_distribution) / self.num_entities
        for j in range(self.num_cells):
            # uniform fluctuation
            for i in range(self.num_subpopulation):
                self.potential[i,j] += self.flucturations[i] * (2*random.random() - 1.0)

    def update_flucturation(self):
        new_subpopulation_distribution = self.subpopulation_distribution.copy()
        for x in range(self.num_cells):
            for j in range(self.num_subpopulation):
                moves = {-1: 0, 1: 0, 0: 0}
                
                for i in [-1, 1]:
                    next_x = x + i
                    if next_x < 0 or next_x >= self.num_cells:
                        continue

                    potential_diff = self.potential[j][next_x] - self.potential[j][x]
                    prob_move = max(0, potential_diff)
                    moves[i] = prob_move
                
                # normalize
                moves[0] = max(0, 1 - moves[-1] - moves[1])

                # this is useful when the sum of moves is not 1
                scale_factor = sum(moves.values())

                # the old subpopulation distribution
                num_entities = self.subpopulation_distribution[j][x]
                for k in range(num_entities):
                    
                    next_x = x
                    rand = random.random() * scale_factor
                    if rand < moves[-1]:
                        next_x = x - 1
                    elif rand < moves[1] + moves[-1]:
                        next_x = x + 1

                    if next_x < 0 or next_x >= self.num_cells:
                        print(f"next_x: {next_x}, x: {x}, moves: {moves}")
                        continue

                    new_subpopulation_distribution[j][x] -= 1
                    new_subpopulation_distribution[j][next_x] += 1
                    
        self.subpopulation_distribution = new_subpopulation_distribution

    def update_diffusion(self):
        new_subpopulation_distribution = self.subpopulation_distribution.copy()
        for j in range(self.num_subpopulation):
            if random.random() < self.diffusion_velocity[j]:
                for x in range(self.num_cells):
                        if x + self.diffusion_direction[j] < 0 or x + self.diffusion_direction[j] >= self.num_cells:
                            continue
                        num_entities = self.subpopulation_distribution[j][x]
                        new_subpopulation_distribution[j][x+self.diffusion_direction[j]] += num_entities
                        new_subpopulation_distribution[j][x] -= num_entities
                                
        self.subpopulation_distribution = new_subpopulation_distribution

    def update(self):
        self.update_flucturation()
        self.update_diffusion()
        return
    
