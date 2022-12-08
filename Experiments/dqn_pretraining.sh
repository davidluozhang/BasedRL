#!/bin/zsh
echo "===== Debugging script for pretraining poacher agent in new environment ====="
echo "===== dqn_fix_4, poacher survival reward, ranger first, short epochs, fast alternation, starting ====="
#python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 100 --step-per-epoch 10000 --lr 5e-5 --logdir "dqn_pretrained/ranger"
python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 100 --step-per-epoch 10000 --lr 1e-5 --logdir "dqn_pretrained/poacher"
#python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 5 --step-per-epoch 10000 --lr .01 --logdir "dqn_fix_4/ranger"
#for i in {1..100}
#do
#echo "===== Iteration $i ====="
#python ~/BasedRL/main.py --agent-id 1 --agent-learn "dqn" --epoch 1 --step-per-epoch 1000 --lr 5e-5 --agent-opponent "dqn" --resume-path "dqn_fix_4/ranger/gsg/dqn/policy.pth" --opponent-path "dqn_fix_4/poacher/gsg/dqn/policy.pth" --logdir "dqn_fix_4/ranger" --eps-train 0.15
#python ~/BasedRL/main.py --agent-id 0 --agent-learn "dqn" --epoch 1 --step-per-epoch 1000 --lr 5e-5 --agent-opponent "dqn" --resume-path "dqn_fix_4/poacher/gsg/dqn/policy.pth" --opponent-path "dqn_fix_4/ranger/gsg/dqn/policy.pth" --logdir "dqn_fix_4/poacher" --eps-train 0.15
#done
#echo "===== dqn_fix_4, poacher survival reward, ranger first, short epochs, fast alternation, finished ====="
