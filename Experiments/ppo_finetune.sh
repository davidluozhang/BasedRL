#!/bin/zsh
echo "===== Debugging script for pretraining poacher agent in new environment ====="
echo "===== ppo_finetune, poacher survival reward, ranger first, short epochs, fast alternation, starting ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "ppo" --epoch 1 --step-per-epoch 1000 --lr 1e-2 --agent-opponent "ppo" --resume-path "ppo_pretrained/ranger/gsg/ppo/policy.pth" --opponent-path "ppo_pretrained/poacher/gsg/ppo/policy.pth" --logdir "ppo_fix_4/ranger" --eps-train 0.15
python ~/BasedRL/main.py --agent-id 0 --agent-learn "ppo" --epoch 1 --step-per-epoch 1000 --lr 1e-2 --agent-opponent "ppo" --resume-path "ppo_pretrained/poacher/gsg/ppo/policy.pth" --opponent-path "ppo_pretrained/ranger/gsg/ppo/policy.pth" --logdir "ppo_fix_4/poacher" --eps-train 0.15
for i in {1..100}
do
echo "===== Iteration $i ====="
python ~/BasedRL/main.py --agent-id 1 --agent-learn "ppo" --epoch 1 --step-per-epoch 1000 --lr 5e-2 --agent-opponent "ppo" --resume-path "ppo_fix_4/ranger/gsg/ppo/policy.pth" --opponent-path "ppo_fix_4/poacher/gsg/ppo/policy.pth" --logdir "ppo_fix_4/ranger" --iter $i --eps-train 0.15
echo "===== Training Poacher Now ====="
python ~/BasedRL/main.py --agent-id 0 --agent-learn "ppo" --epoch 1 --step-per-epoch 1000 --lr 5e-2 --agent-opponent "ppo" --resume-path "ppo_fix_4/poacher/gsg/ppo/policy.pth" --opponent-path "ppo_fix_4/ranger/gsg/ppo/policy.pth" --logdir "ppo_fix_4/poacher" --iter $i --eps-train 0.15
done
echo "===== ppo_finetune, poacher survival reward, ranger first, short epochs, fast alternation, finished ====="
