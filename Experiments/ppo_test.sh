#!/bin/zsh
echo "===== Debugging script for pretraining poacher agent in new environment ====="
echo "===== ppo_test, poacher survival reward, ranger first, short epochs, fast alternation, starting ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "ppo" --epoch 1 --step-per-epoch 1000 --lr .01 --logdir "ppo_test/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "ppo" --epoch 5 --step-per-epoch 10000 --lr .01 --logdir "ppo_test/poacher"
#python ~/BasedRL/main.py --agent-id 1 --agent-learn "ppo" --epoch 5 --step-per-epoch 10000 --lr .01 --logdir "ppo_test/ranger"
for i in {1..20}
do
echo "===== Iteration $i ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "ppo" --epoch 1 --step-per-epoch 10000 --lr .001 --agent-opponent "ppo" --resume-path "ppo_test/ranger/gsg/ppo/policy.pth" --opponent-path "ppo_test/poacher/gsg/ppo/policy.pth" --logdir "ppo_test/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "ppo" --epoch 1 --step-per-epoch 10000 --lr .001 --agent-opponent "ppo" --resume-path "ppo_test/poacher/gsg/ppo/policy.pth" --opponent-path "ppo_test/ranger/gsg/ppo/policy.pth" --logdir "ppo_test/poacher"
done
echo "===== ppo_test, poacher survival reward, ranger first, short epochs, fast alternation, finished ====="
