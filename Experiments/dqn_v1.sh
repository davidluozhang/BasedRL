#!/bin/zsh
echo "===== dqn_alternating_5, ranger first, starting ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 1 --lr .1 --logdir "dqn_alternating_5/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 1 --lr .1 --logdir "dqn_alternating_5/poacher"
for i in {1..3}
do
echo "===== Iteration $i ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 6 --lr .1 --agent-opponent "dqn" --resume-path "dqn_alternating_5/ranger/gsg/dqn/policy.pth" --opponent-path "dqn_alternating_5/poacher/gsg/dqn/policy.pth" --logdir "dqn_alternating_5/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 6 --lr .1 --agent-opponent "dqn" --resume-path "dqn_alternating_5/poacher/gsg/dqn/policy.pth" --opponent-path "dqn_alternating_5/ranger/gsg/dqn/policy.pth" --logdir "dqn_alternating_5/poacher"
done
echo "===== dqn_alternating_5, ranger first, finished ====="
