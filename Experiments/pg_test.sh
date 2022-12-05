#!/bin/zsh
echo "===== Debugging script for pretraining poacher agent in new environment ====="
echo "===== dqn_reinforce_3, poacher survival reward, ranger first, short epochs, fast alternation, starting ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "reinforce" --epoch 1 --step-per-epoch 1000 --lr .01 --logdir "dqn_reinforce_3/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "reinforce" --epoch 5 --step-per-epoch 10000 --lr .01 --logdir "dqn_reinforce_3/poacher"
python ~/BasedRL/main.py --agent-id 1 --agent-learn "reinforce" --epoch 5 --step-per-epoch 10000 --lr .01 --logdir "dqn_reinforce_3/ranger"
for i in {1..20}
do
echo "===== Iteration $i ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "reinforce" --epoch 1 --step-per-epoch 3000 --lr .001 --agent-opponent "reinforce" --resume-path "dqn_reinforce_3/ranger/gsg/reinforce/policy.pth" --opponent-path "dqn_reinforce_3/poacher/gsg/reinforce/policy.pth" --logdir "dqn_reinforce_3/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "reinforce" --epoch 1 --step-per-epoch 3000 --lr .001 --agent-opponent "reinforce" --resume-path "dqn_reinforce_3/poacher/gsg/reinforce/policy.pth" --opponent-path "dqn_reinforce_3/ranger/gsg/reinforce/policy.pth" --logdir "dqn_reinforce_3/poacher"
done
echo "===== dqn_reinforce_3, poacher survival reward, ranger first, short epochs, fast alternation, finished ====="
