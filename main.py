"""This is a full example of using Tianshou with MARL to train agents, complete with argument parsing (CLI) and logging.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import argparse
import os
from copy import deepcopy
from typing import Optional, Union, Tuple
import faulthandler

import gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, PGPolicy, PPOPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from masked_ppo import MaskedPPO

# updated environment
# from GSGEnvironment.gsg_environment import gsg_environment_v2 as gsg_environment
# from GSGEnvironment.gsg_environment import gsg_environment_v3 as gsg_environment
from GSGEnvironment.gsg_environment import gsg_pygame as gsg_environment
faulthandler.enable()

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.07)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--iter", type=str, default="", help="jank")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--agent-learn",
        default='random',
        help="what algorithm do we use for agent we are currently training"
    )
    parser.add_argument(
        "--agent-opponent",
        default='random',
        help="what algorithm do we use for opponent"
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=0,
        help="which agent to train - 0 for poacher, 1 for ranger"
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )

    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--eval_only", action='store_true',
    )
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: Union[BasePolicy, str] = None,
    agent_opponent: Union[BasePolicy, str] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = (
        observation_space["observation"].shape or observation_space["observation"].n
    )
    args.action_shape = env.action_space.shape or env.action_space.n

    if agent_learn == 'dqn':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)
        agent_learn = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
            lr_scheduler=scheduler,
        )
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    elif agent_learn == 'reinforce':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)
        agent_learn = PGPolicy(
            net,
            optim,
            Categorical,
            args.gamma,
        ) # TODO add support for additional hyperparameters
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    elif agent_learn == 'ppo':
        # model
        actor_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        )

        critic_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        )

        actor = Actor(
            preprocess_net=actor_net,
            action_shape=5,
        ).to(args.device)

        critic = Critic(
            preprocess_net=critic_net,
            device=args.device,
        ).to(args.device)

        # TODO there might be some manual weight initialization strategies that work better here
        if optim is None:
            optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

        agent_learn = MaskedPPO(
            actor,
            critic,
            optim,
            Categorical,
            args.gamma,
        ) # TODO add support for additional hyperparameters
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    elif agent_learn == 'random':
        agent_learn = RandomPolicy()

    if agent_opponent == 'dqn':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)

        agent_opponent = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
            lr_scheduler=scheduler,
        )
        if args.resume_path:
            # print(f"Loading pretrained opponent DQN model from {args.opponent_path}")
            agent_opponent.load_state_dict(torch.load(args.opponent_path))

    elif agent_opponent == 'reinforce':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_opponent = PGPolicy(
            net,
            optim,
            Categorical,
            args.gamma,
        ) # TODO add support for additional hyperparameters
        if args.resume_path:
            agent_opponent.load_state_dict(torch.load(args.opponent_path))

    elif agent_opponent == 'ppo':
        # model
        actor_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        )

        critic_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        )

        actor = Actor(
            preprocess_net=actor_net,
            action_shape=5,
        ).to(args.device)

        critic = Critic(
            preprocess_net=critic_net,
            device=args.device,
        ).to(args.device)

        # TODO there might be some manual weight initialization strategies that work better here
        if optim is None:
            optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

        agent_opponent = MaskedPPO(
            actor,
            critic,
            optim,
            Categorical,
            args.gamma,
        ) # TODO add support for additional hyperparameters
        if args.resume_path:
            agent_opponent.load_state_dict(torch.load(args.opponent_path))

    elif agent_opponent == 'random':
        agent_opponent = RandomPolicy()

    if args.agent_id == 0:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    #import pdb
    #pdb.set_trace()
    return policy, optim, env.agents


def get_env(render_mode=None):
    return PettingZooEnv(gsg_environment.env(render_mode=render_mode))

def train_agent(
    args: argparse.Namespace = get_args(),
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # set learning_algorithms
    agent_learn = args.agent_learn
    agent_opponent = args.agent_opponent

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    #import pdb
    #pdb.set_trace()
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "gsg", args.agent_learn, args.iter)
    print(f"tensorboard log path: {log_path}")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, train_interval=100)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "gsg", args.agent_learn, "policy.pth"
            )
        print(f"Model Save Path: {model_save_path}")
        torch.save(
            policy.policies[agents[args.agent_id]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents[args.agent_id]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, args.agent_id]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=None,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy.policies[agents[args.agent_id]], policy.policies[agents[args.agent_id-1]]


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])

    if not agent_learn:
        agent_learn = args.agent_learn
    if not agent_opponent:
        agent_opponent = args.agent_opponent

    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    policy.policies[agents[args.agent_id]].set_eps(args.eps_test)
    #policy.policies[agents[args.agent_id-1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)

    if args.eval_only:
        #collector = Collector(policy, env, exploration_noise=False)
        result = collector.collect(n_episode=1, render=0.05)
        # result = collector.collect(n_episode=1)
    else:
        #collector = Collector(policy, env, exploration_noise=True)
        result = collector.collect(n_episode=100)
    #import pdb
    #pdb.set_trace()
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    torch.set_default_dtype(torch.float32)
    args = get_args()
    if args.eval_only:
        assert args.resume_path is not None
        assert args.opponent_path is not None
        watch(args)
    else:
        result, agent_learned, agent_opp = train_agent(args)
        watch(args, agent_learned, agent_opp)
