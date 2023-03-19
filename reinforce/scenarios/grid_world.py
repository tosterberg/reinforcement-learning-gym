import collections
import numpy as np
import random


class GridWorld:
    """
        GridWorld class
            A grid_world class implementation creates a 2D environment with up, down, left, right
            actions, and reward based on state (position) on the board
        interface
            actions - number of available moves denoting direction
        variables
            label - scenario name
        methods
            reset - resets the grid worlds action and state values to new values
            reward - return reward value given action choice, for current state
    """
    def __init__(self, grid_x=4, grid_y=4, r_values=[10, 5],
                 r_val_base=0, r_val_out=-1, **kwargs):
        self.label = "GridWorld"
        self.actions = {
            0: {
                "name": "up",
                "transition": (0, -1)
            },
            1: {
                "name": "down",
                "transition": (0, 1)
            },
            2: {
                "name": "left",
                "transition": (-1, 0)
            },
            3: {
                "name": "down",
                "transition": (1, 0)
            }
        }
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.r_values = r_values
        self.r_val_base = r_val_base
        self.r_val_out = r_val_out
        self.states = self.make_hash()
        self.current = tuple()
        self.reset()

    def __str__(self):
        return f'{self.label}: actions={len(self.actions.keys())}, grid_size={self.grid_x*self.grid_y},' \
               f' r_values={self.r_values}, base_r={self.r_val_base}, out_r={self.r_val_out}'

    def reward(self, action):
        x, y = self.current
        r = self.states[x][y][action]
        self.transition_state(action)
        return r

    def make_hash(self):
        return collections.defaultdict(self.make_hash)

    def state(self):
        return self.current

    def transition_state(self, action):
        if self.is_off_grid(self.current, action):
            return
        x, y = self.current
        x += self.actions[action]["transition"][0]
        y += self.actions[action]["transition"][1]
        self.current = (x, y)


    def init_grid(self):
        available_spots = list(range(self.grid_x * self.grid_y))
        start = random.choice(available_spots)
        self.current = (start // self.grid_y, start // self.grid_x)
        for reward in self.r_values:
            location = random.choice(available_spots)
            self.generate_node(location, reward)
            available_spots = available_spots[:location] + available_spots[location+1:]
        for location in available_spots:
            self.generate_node(location, self.r_val_base)

    def generate_node(self, flat_index, reward):
        x = flat_index // self.grid_y
        y = flat_index // self.grid_x
        for key in self.actions.keys():
            self.states[x][y][key] = -1 if self.is_off_grid((x, y), key) else reward

    def is_off_grid(self, node, action):
        x, y = node
        x += self.actions[action]["transition"][0]
        y += self.actions[action]["transition"][1]
        if x < 0 or x >= self.grid_x:
            return True
        if y < 0 or y >= self.grid_y:
            return True
        return False

    def reset(self):
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                for k in self.actions.keys():
                    self.states[i][j][k] = self.r_val_base
        self.init_grid()
