# AttackDefendRL
Fall 22 CIS 520 Final Project

## Installation & Quickstart

The current environment requires PyTorch 1.13 and PettingZoo 1.22.2, which can be installed via the provided Conda environment file. 

```
git clone https://github.com/davidluozhang/AttackDefendRL
cd AttackDefendRL
conda env create -f environment.yml
```

You should then be able to test your installation by running `test_zoo.py`, which is a simple `pettingzoo` example taken from the docs.

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_zoo.py
```
if your Mac has an M1 chip (based),
```
python test_zoo.py
```
if your name is Alyssa.

A few caveats:
- The environment has only been tested with Python 3.10, use Python 3.9 at your own risk
- The default `pip install supersuit` requires a different `pymunk` version, but only `pymunk==6.3.0` will work because of a jank thing with callbacks and Apple security. 
