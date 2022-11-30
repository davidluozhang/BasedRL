#!/bin/zsh
echo "===== dqn_fast_2, poacher survival reward, ranger first, short epochs, fast alternation, starting ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 1 --step-per-epoch 1000 --lr .01 --logdir "dqn_fast_2/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 1 --step-per-epoch 1000 --lr .01 --logdir "dqn_fast_2/poacher"
for i in {1..20}
do
echo "===== Iteration $i ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 1 --step-per-epoch 1000 --lr .01 --agent-opponent "dqn" --resume-path "dqn_fast_2/ranger/gsg/dqn/policy.pth" --opponent-path "dqn_fast_2/poacher/gsg/dqn/policy.pth" --logdir "dqn_fast_2/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 1 --step-per-epoch 1000 --lr .01 --agent-opponent "dqn" --resume-path "dqn_fast_2/poacher/gsg/dqn/policy.pth" --opponent-path "dqn_fast_2/ranger/gsg/dqn/policy.pth" --logdir "dqn_fast_2/poacher"
done
echo "===== dqn_fast_2, poacher survival reward, ranger first, short epochs, fast alternation, finished ====="
