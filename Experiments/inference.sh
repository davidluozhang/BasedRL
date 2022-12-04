RESUME_PATH=Experiments/dqn_fast_2_converged/ranger/gsg/dqn/policy.pth
OPPONENT_PATH=Experiments/dqn_fast_2_converged/poacher/gsg/dqn/policy.pth
LEARN_ALG=dqn
OPP_ALG=dqn

python main.py --eval_only --resume-path ${RESUME_PATH} --opponent-path ${OPPONENT_PATH} --agent-learn ${LEARN_ALG} --agent-opponent ${OPP_ALG}