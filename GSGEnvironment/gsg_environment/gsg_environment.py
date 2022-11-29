import functools
import random
from copy import copy

import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium import spaces

from pettingzoo.utils.env import ParallelEnv

# Global vars
NUM_ROWS = 4
NUM_COLS = 4
POACHER_INIT_X = 0
POACHER_INIT_Y = 0
POACHER_NUM_TRAPS = 3
POACHER_CATCH_REWARD = 0.5
POACHER_CAUGHT_PENALTY = 20

RANGER_INIT_X = 3
RANGER_INIT_Y = 3
RANGER_TRAP_REWARD = 0.5
RANGER_CATCH_REWARD = 20
TIMEOUT = 100


class CustomEnvironment(ParallelEnv):
    def __init__(self):
        self.poacher_x = None
        self.poacher_y = None
        self.poacher_traps = None
        self.ranger_x = None
        self.ranger_y = None
        self.timestep = None
        self.observation_spaces = {
            "ranger": spaces.Dict({
                "observation": spaces.Box(
                    low=0, high=1, shape=(NUM_ROWS, NUM_COLS, 2), dtype=np.int8
                ),
                "action_mask": spaces.Discrete(4),
            }),
            "poacher": spaces.Dict({
                "observation": spaces.Box(
                    low=0, high=1, shape=(NUM_ROWS, NUM_COLS, 2), dtype=np.int8
                ),
                "action_mask": spaces.Discrete(5),
            }),
        }
        self.grid = []
        for i in range(NUM_ROWS):
            row = [{'animal_density': random.random(), 'has_trap': False, 'ranger_vis': False, 'poacher_vis': False} for
                   _ in range(NUM_COLS)]
            self.grid.append([row])
        self.possible_agents = ["poacher", "ranger"]

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.poacher_x = POACHER_INIT_X
        self.poacher_y = POACHER_INIT_Y
        self.poacher_traps = POACHER_NUM_TRAPS

        self.ranger_X = RANGER_INIT_X
        self.ranger_y = RANGER_INIT_Y

        observation = []  # this might be rlly jank

        for i in range(NUM_ROWS):
            observation.append(
                [{'ranger_vis': self.grid[i][j]['ranger_vis'], 'poacher_vis': self.grid[i][j]['poacher_vis']} for j in
                 range(NUM_COLS)])

        observation = (observation)  # ToDo: What is this?

        observations = {
            "ranger": {"observation": observation, "action_mask": [1, 0, 0, 1]},
            "poacher": {"observation": observation, "action_mask": [0, 1, 1, 0, 1]},
        }
        return observations

    """
    Actions: two params of poacher and ranger 
    """

    def step(self, actions):
        ranger_action = actions["ranger"]
        poacher_action = actions["poacher"]

        # 0, 1, 2, 3
        # ToDo: Add an option for no movement?
        if ranger_action == "LEFT" and self.ranger_x > 0:
            self.ranger_x -= 1
        elif ranger_action == "RIGHT" and self.ranger_x < NUM_ROWS - 1:
            self.ranger_x += 1
        elif ranger_action == "UP" and self.ranger_y > 0:
            self.ranger_y -= 1
        elif ranger_action == "BOTTOM" and self.ranger_y < NUM_COLS - 1:
            self.ranger_y += 1

        if poacher_action == "LEFT" and self.poacher_x > 0:
            self.poacher_x -= 1
        elif poacher_action == "RIGHT" and self.poacher_x < NUM_ROWS - 1:
            self.poacher_x += 1
        elif poacher_action == "UP" and self.poacher_y > 0:
            self.poacher_y -= 1
        elif poacher_action == "BOTTOM" and self.poacher_y < NUM_COLS - 1:
            self.poacher_y += 1
        elif poacher_action == "PLACE" and self.poacher_traps > 0:
            self.grid[self.poacher_y][self.poacher_x]['has_trap'] = True
            self.poacher_traps -= 1

        poacher_action_mask = np.ones(5)
        if self.poacher_x == 0:
            poacher_action_mask[0] = 0  # block left movement
        elif self.poacher_x == NUM_COLS - 1:
            poacher_action_mask[1] = 0  # block right movement
        if self.poacher_y == 0:
            poacher_action_mask[2] = 0  # block down movement
        elif self.poacher_y == NUM_ROWS - 1:
            poacher_action_mask[3] = 0  # block up movement
        if self.grid[self.poacher_y][self.poacher_x]['has_trap']:
            poacher_action_mask[4] = 0  # can't place trap twice
        if self.poacher_traps == 0:
            poacher_action_mask[4] = 0  # out of traps

        ranger_action_mask = np.ones(4)
        if self.ranger_x == 0:
            ranger_action_mask[0] = 0  # block left movement
        elif self.ranger_x == NUM_COLS - 1:
            ranger_action_mask[1] = 0  # block right movement
        if self.ranger_y == 0:
            ranger_action_mask[2] = 0  # block down movement
        elif self.ranger_y == NUM_ROWS - 1:
            ranger_action_mask[3] = 0  # block up movement

        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        # terminate if ranger catches poacher
        if self.poacher_x == self.ranger_x and self.poacher_y == self.ranger_y:
            rewards = {"poacher": POACHER_CAUGHT_PENALTY, "ranger": RANGER_CATCH_REWARD}
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Reward ranger if it picks up a trap: 
        if self.grid[self.ranger_y][self.ranger_x]['has_trap']:
            self.grid[self.ranger_y][self.ranger_x]['has_trap'] = False
            rewards["ranger"] += RANGER_TRAP_REWARD

        # Reward poacher for placed traps
        rewards["poacher"] += self.get_poacher_reward()

        truncations = {"poacher": False, "ranger": False}
        if self.timestep > TIMEOUT:
            truncations = {"ranger": True, "poacher": True}
            self.agents = []
        self.timestep += 1

        observation = []  # this might be rlly jank

        for i in range(NUM_ROWS):
            observation.append(
                [{'ranger_vis': self.grid[i][j]['ranger_vis'], 'poacher_vis': self.grid[i][j]['poacher_vis']} for j in
                 range(NUM_COLS)])

        observation = (observation)

        observations = {
            "ranger": {"observation": observation, "action_mask": ranger_action_mask},
            "poacher": {"observation": observation, "action_mask": poacher_action_mask},
        }

        # Get dummy infos TODO maybe add something here
        infos = {"ranger": {}, "poacher": {}}

        return observations, rewards, terminations, truncations, infos

    # ToDo: Penalize the poacher for how many traps it puts down to prevent spamming
    def get_poacher_reward(self):
        poacher_trap_reward = 0
        for i in range(NUM_ROWS):
            for j in range(NUM_COLS):
                if self.grid[i][j]['has_trap']:
                    poacher_trap_reward += POACHER_CATCH_REWARD if random.random() < self.grid[i][j][
                        'animal_density'] else 0
        return poacher_trap_reward

    def render(self):
        grid = np.zeros((NUM_ROWS, NUM_COLS))
        grid[self.ranger_y, self.ranger_x] = "R"
        grid[self.poacher_y, self.poacher_x] = "P"
        for y in range(NUM_COLS):
            for x in range(NUM_ROWS):
                if self.grid[y][x]['has_trap']:
                    grid[y, x] = "T" if grid[y, x] == 0 else grid[y, x] + "T"
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        return MultiDiscrete([7 * 7 - 1] * 3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
