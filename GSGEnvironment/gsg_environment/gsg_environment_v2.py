import functools
import random
from copy import copy

import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium import spaces

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# Global vars
NUM_ROWS = 4
NUM_COLS = 4
POACHER_INIT_X = 0
POACHER_INIT_Y = 0
POACHER_NUM_TRAPS = 5
POACHER_CATCH_REWARD = 2 
POACHER_CAUGHT_PENALTY = -20

RANGER_INIT_X = 3
RANGER_INIT_Y = 3
RANGER_TRAP_REWARD = 1
RANGER_CATCH_REWARD = 20
RANGER_TIME_PENALTY = 0.3
TIMEOUT = 300
FOOTSTEP_VANISH_TIME = 10 
OBSTACLES = [(0, 1), (3, 3), (2, 2)]
PROXIMITY_DISTANCE = 2


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = CustomEnvironment(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class CustomEnvironment(AECEnv):
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "gsg_environment",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, render_mode="ansi"):
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
                "action_mask": spaces.Discrete(5),
            }),
            "poacher": spaces.Dict({
                "observation": spaces.Box(
                    low=0, high=1, shape=(NUM_ROWS, NUM_COLS, 2), dtype=np.int8
                ),
                "action_mask": spaces.Discrete(5),
            }),
        }

        self.action_spaces = {
            "ranger": spaces.Discrete(5),
            "poacher": spaces.Discrete(5),
        }

        self.grid = []
        random.seed(123)
        for i in range(NUM_ROWS):
            row = [{'animal_density': random.random(), 'has_trap': False, 'ranger_vis': [], 'poacher_vis': [], 'obstacle': False} for
                   _ in range(NUM_COLS)]
            self.grid.append(row)

        ##TODO: POPULATE OBSTACLES W/ VARIABLE 

        self.possible_agents = ["poacher", "ranger"]
        self.agents = self.possible_agents[:]

        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}  # TODO maybe actually use infos

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.render_mode = render_mode
        self.timestep = 0

    def observe(self, agent):
        # board_vals = np.array(self.grid).reshape(NUM_ROWS, NUM_COLS)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        # cur_p_board = np.equal(board_vals, cur_player + 1)
        # opp_p_board = np.equal(board_vals, opp_player + 1)
        def check_obstacles(y, x, action_mask):
            for i, j in OBSTACLES: #i is x, j is y 
                if x - 1 == i:
                    action_mask[0] = 0 #block left movement 
                if x + 1 == i: 
                    action_mask[1] = 0 #block right mvmt 
                if y - 1 == j: 
                    action_mask[2] = 0 #block down mvmt 
                if y + 1 == j:
                    action_mask[3] = 0 #block up mvmt 
            return action_mask 

        # observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        action_mask = np.ones(5) #current actions you can take 
        if agent == 'poacher':
            if self.poacher_x == 0:
                action_mask[0] = 0  # block left movement
            elif self.poacher_x == NUM_COLS - 1:
                action_mask[1] = 0  # block right movement
            if self.poacher_y == 0:
                action_mask[2] = 0  # block down movement
            elif self.poacher_y == NUM_ROWS - 1:
                action_mask[3] = 0  # block up movement
            if self.grid[self.poacher_y][self.poacher_x]['has_trap']:
                action_mask[4] = 0  # can't place trap twice
            if self.poacher_traps == 0:
                action_mask[4] = 0  # out of traps
            action_mask = check_obstacles(self.poacher_y, self.poacher_x, action_mask)

        if agent == 'ranger':
            if self.ranger_x == 0:
                action_mask[0] = 0  # block left movement
            elif self.ranger_x == NUM_COLS - 1:
                action_mask[1] = 0  # block right movement
            if self.ranger_y == 0:
                action_mask[2] = 0  # block down movement
            elif self.ranger_y == NUM_ROWS - 1:
                action_mask[3] = 0  # block up movement
            # the ranger cannot place traps
            action_mask[-1] = 0
            action_mask = check_obstacles(self.ranger_y, self.ranger_x, action_mask)

        # ranger_steps = []  # this might be rlly jank
        # poacher_steps = []  # this might be rlly jank

        # for i in range(NUM_ROWS):
        #     ranger_steps.append(
        #         [self.grid[i][j]['ranger_vis'] for j in range(NUM_COLS)]
        #     )
        #     poacher_steps.append(
        #         [self.grid[i][j]['poacher_vis'] for j in range(NUM_COLS)]
        #     )
        # observation = np.stack([np.array(ranger_steps), np.array(poacher_steps)], axis=2).astype(np.int8)
        # # observation = (observation)
        observation = self.get_observations()
        observations = {
            "ranger": {"observation": observation, "action_mask": np.zeros(5)},
            "poacher": {"observation": observation, "action_mask": np.zeros(5)},
        }

        if agent == 'poacher':
            observations["poacher"]["action_mask"] = action_mask

        elif agent == 'ranger':
            observations["ranger"]["action_mask"] = action_mask

        return {"observation": observation, "action_mask": action_mask}

    def reset(self, seed=None, return_info=False, options=None):

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}  # TODO maybe actually use infos
        self.poacher_x = POACHER_INIT_X
        self.poacher_y = POACHER_INIT_Y
        self.poacher_traps = POACHER_NUM_TRAPS

        self.ranger_x = RANGER_INIT_X
        self.ranger_y = RANGER_INIT_Y

        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()
        self.timestep = 0
        #clear grid 
        for i in range(NUM_ROWS):
            for j in range(NUM_COLS):
                self.grid[i][j]['has_trap'] = False
                self.grid[i][j]['ranger_vis'] = []
                self.grid[i][j]['poacher_vs'] = []
        '''
        observation = []  # this might be rlly jank

        for i in range(NUM_ROWS):
            observation.append(
                [{'ranger_vis': self.grid[i][j]['ranger_vis'], 'poacher_vis': self.grid[i][j]['poacher_vis']} for j in
                 range(NUM_COLS)])

        # observation = (observation)  # ToDo: What is this?

        observations = {
            "ranger": {"observation": observation, "action_mask": [1, 0, 0, 1, 0]},
            "poacher": {"observation": observation, "action_mask": [0, 1, 1, 0, 1]},
        }
        return observations
        '''

    """
    Actions: two params of poacher and ranger
    """

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            return self._was_dead_step(action)

        # 0, 1, 2, 3
        # ToDo: Add an option for no movement?
        if self.agent_selection == 'ranger':
            if action == 0 and self.ranger_x > 0:
                self.ranger_x -= 1
            elif action == 1 and self.ranger_x < NUM_ROWS - 1:
                self.ranger_x += 1
            elif action == 2 and self.ranger_y > 0:
                self.ranger_y -= 1
            elif action == 3 and self.ranger_y < NUM_COLS - 1:
                self.ranger_y += 1
            self.grid[self.ranger_y][self.ranger_x]['ranger_vis'].append(self.timestep)

        elif self.agent_selection == 'poacher':
            if action == 0 and self.poacher_x > 0:
                self.poacher_x -= 1
            elif action == 1 and self.poacher_x < NUM_ROWS - 1:
                self.poacher_x += 1
            elif action == 2 and self.poacher_y > 0:
                self.poacher_y -= 1
            elif action == 3 and self.poacher_y < NUM_COLS - 1:
                self.poacher_y += 1
            elif action == 4 and self.poacher_traps > 0:
                self.grid[self.poacher_y][self.poacher_x]['has_trap'] = True
                self.poacher_traps -= 1
            self.grid[self.poacher_y][self.poacher_x]['poacher_vis'].append(self.timestep)

        next_agent = self._agent_selector.next()

        # terminate if ranger catches poacher
        if self.poacher_x == self.ranger_x and self.poacher_y == self.ranger_y:
            self.rewards['poacher'] = POACHER_CAUGHT_PENALTY
            self.rewards['ranger'] = RANGER_CATCH_REWARD
            # print(f"Caught after {self.timestep} steps")
            self.terminations = {a: True for a in self.agents}

        # Reward ranger if it picks up a trap:
        if self.grid[self.ranger_y][self.ranger_x]['has_trap']:
            self.grid[self.ranger_y][self.ranger_x]['has_trap'] = False
            self.rewards["ranger"] += RANGER_TRAP_REWARD

        # Reward poacher for placed traps and survival
        self.rewards["poacher"] += self.get_poacher_reward()
        # self.rewards["poacher"] += RANGER_TIME_PENALTY

        # Penalize ranger for time delay and caught animals
        self.rewards["ranger"] -= self.get_poacher_reward()
        self.rewards["ranger"] -= RANGER_TIME_PENALTY
        self.timestep += 1

        # print(f"Timestep {self.timestep}: Poacher Reward {self.rewards['poacher']}")
        # print(f"Timestep {self.timestep}: Poacher Location {self.poacher_x} {self.poacher_y}")
        # print(f"Timestep {self.timestep}: Ranger Reward {self.rewards['ranger']}")
        # print(f"Timestep {self.timestep}: Ranger Location {self.ranger_x} {self.ranger_y}")

        if self.timestep > TIMEOUT:
            self.truncations = {a: True for a in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        # if self.render_mode == "human" and self.agent_selection == 'poacher':
        #    self.render()
    
    def get_observations(self, proximity = False):
        #checks proximity of grid cell with agent 
        def distance_apart(x, y, agent_x, agent_y, maxDistance):
            return abs(x-agent_x) <= maxDistance and abs(y-agent_y) <= maxDistance

        def check_footstep(y, x, proximity=False, isRanger = True):
            agent = 'ranger_vis' if isRanger else 'poacher_vis'
            #get coordinates of the agent 
            agent_x = self.ranger_x if isRanger else self.poacher_x 
            agent_y = self.ranger_y if isRanger else self.poacher_y
            if self.grid[y][x][agent].length > 0:
                if self.timestep - self.grid[y][x][agent][-1] < FOOTSTEP_VANISH_TIME:
                    if proximity and not distance_apart(x, y, agent_x, agent_y, PROXIMITY_DISTANCE):
                        return 0 ##if proximity is enabled and it's too far out of range 
                    return 1 
            return 0
        ranger_steps = []
        poacher_steps = []
        for i in range(NUM_ROWS):
            ranger_steps.append(
                [check_footstep(i, j, isRanger=True) for j in range(NUM_COLS)]
            )
            poacher_steps.append(
                [check_footstep(i, j, isRanger=False) for j in range(NUM_COLS)]
            )
        return np.stack([np.array(ranger_steps), np.array(poacher_steps)], axis=2).astype(np.int8)
        # observation = (observation)

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
        grid = [['0'] * NUM_COLS for _ in range(NUM_ROWS)]
        grid[self.poacher_y][self.poacher_x] = "P"
        grid[self.ranger_y][self.ranger_x] = "R"
        print("=" * 10)
        for y in range(NUM_COLS):
            for x in range(NUM_ROWS):
                if self.grid[y][x]['has_trap']:
                    grid[y][x] = "T" if grid[y][x] == 0 else grid[y][x] + "T"
            print(f"{grid[y]} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
