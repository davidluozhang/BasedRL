import gym
from tianshou.policy import RandomPolicy
import os
from ..GSGEnvironment.gsg_environment.gsg_environment import CustomEnvironment

if __name__ == '__main__':

    env = CustomEnvironment()

    model = RandomPolicy(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1
    )
    model.learn(total_timesteps=100000)