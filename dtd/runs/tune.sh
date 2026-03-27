#!/bin/bash
# tuning parameters: lmb.lr / regularization coef / weightunet.lr

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# On macOS, force CPU backend to avoid noisy TPU backend probing.
if [[ "$(uname -s)" == "Darwin" ]]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi


UNIXTIME=$(date +%s)
AGENT_CLASS="ppo"
ENV_NAME="hopper"
MAX_BUDGET="1e7"
MIN_BUDGET="1e5"

TD="dtd" # baseline / naive / dtd
NOISE_LVL="0.01"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')


echo "Running experiment with:"
echo "ENV_NAME: $ENV_NAME"
echo "NOISE_LVL: $NOISE_LVL"
echo "TD: $TD"
echo "=============================="


mkdir -p "${SCRIPT_DIR}/hpo_results/${ENV_NAME}"
cd "${REPO_ROOT}" || exit 1

"${PYTHON_BIN}" -m dtd.${AGENT_CLASS}.tune --multirun \
    hydra.run.dir="${REPO_ROOT}/dtd/configs/logs/${AGENT_CLASS}/${ENV_NAME}/${TD}/${UNIXTIME}" \
    algorithm=${AGENT_CLASS}_${TD} \
    algorithm.total_timesteps=${MAX_BUDGET} \
    hydra.sweeper.dehb_kwargs.min_budget=${MIN_BUDGET} \
    env.name=${ENV_NAME} \
    env.noise_lvl=${NOISE_LVL} \
    run_time=${UNIXTIME} > "${SCRIPT_DIR}/hpo_results/${ENV_NAME}/${ENV_NAME}_${TD}_noise_lvl${NOISE_LVL_STR}.txt" 2>&1
