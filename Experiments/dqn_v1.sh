#!/bin/zsh
echo "===== dqn_alternating_7, ranger first, starting ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 1 --lr .1 --logdir "dqn_alternating_7/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 1 --lr .1 --logdir "dqn_alternating_7/poacher"
for i in {1..20}
do
echo "===== Iteration $i ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 2 --lr .1 --agent-opponent "dqn" --resume-path "dqn_alternating_7/ranger/gsg/dqn/policy.pth" --opponent-path "dqn_alternating_7/poacher/gsg/dqn/policy.pth" --logdir "dqn_alternating_7/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 2 --lr .1 --agent-opponent "dqn" --resume-path "dqn_alternating_7/poacher/gsg/dqn/policy.pth" --opponent-path "dqn_alternating_7/ranger/gsg/dqn/policy.pth" --logdir "dqn_alternating_7/poacher"
done
echo "===== dqn_alternating_7, ranger first, finished ====="
