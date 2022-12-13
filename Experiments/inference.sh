BASE_PATH=$1
ALG=$2
RESUME_PATH=${BASE_PATH}/ranger/gsg/${2}/policy.pth #Experiments/dqn_fast_2_converged/ranger/gsg/dqn/policy.pth
OPPONENT_PATH=${BASE_PATH}/poacher/gsg/${2}/policy.pth #Experiments/dqn_fast_2_converged/poacher/gsg/dqn/policy.pth
LEARN_ALG=$2
OPP_ALG=$2

python main.py --eval_only --resume-path ${RESUME_PATH} --opponent-path ${OPPONENT_PATH} --agent-learn ${LEARN_ALG} --agent-opponent ${OPP_ALG} --agent-id 1
