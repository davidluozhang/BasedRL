# BasedRL
Fall 22 CIS 520 Final Project

## Introduction & Motivation: 
This repo contains the code for our reinforcement learning project using multi-agent RL in order to model GSG environments. In this work, we've created several custom RL environments to model the patrol route problem for protecting wildlife reserves, running Deep Q-Learning (DQN) and Proximal Policy Optimization (PPO) to identify ranger and poacher strategies.  

## Environment Structure:
The folder GSG contains the various environments that we have implemented:

The below list describes our reward functions and action spaces.

Reward Functions
- Poacher
    - +: Catching animal. 
    - -: Getting caught by ranger
- Rangers: 
    - -: catching animal 
    - +: deactivate traps 
    - +: catch poacher - same square 
Poacher: 4 actions for directions, 1 for setting down trap, 
Ranger: 4 directions (Left, Right, Up, Down)

## Installation & Quickstart

The current environment requires PyTorch 1.13 and PettingZoo 1.22.2, which can be installed via the provided Conda environment file. 

```
git clone https://github.com/davidluozhang/BasedRL
cd BasedRL
conda env create -f environment.yml
```
Because of some jank behavior with the version of `pymunk` that `supersuit` installs, we actually need to install `pymunk==6.3.0`:

```
pip install pymunk==6.3.0
```

You should then be able to test your installation by running `test_zoo.py`, which is a simple `pettingzoo` example taken from the docs.

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_zoo.py
```
if you are running an M1 MacBook.
```
python test_zoo.py
```
if otherwise.

A few caveats:
- The environment has only been tested with Python 3.10, use Python 3.9 at your own risk
- The default `pip install supersuit` requires a different `pymunk` version, but only `pymunk==6.3.0` will work because of a jank thing with callbacks and Apple security. 


## Running Our Code

The `Experiments` folder contains bash scripts useful for reproducing our results, although the `main.py` script contains the main code that creates our agents and handles training, evaluation, and logging.

We use a two-step training process inspired by curriculum learning to help our agents learn and converge effectively. 

To reproduce the DQN results:
```
bash Experiments/dqn_pretraining.sh
```
will run the pretraining procedure and should save the resulting models into `dqn_pretrained/`.

Then, the finetuning procedure is run using:
```
bash Experiments/dqn_finetune.sh
```

To run inference, 

```
bash Experiments/inference.sh /path/to/final/checkpoint <agent method (dqn or ppo)>
```
should give you a neat visualization of one episode played using the final agents.
