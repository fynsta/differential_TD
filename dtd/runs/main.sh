#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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


UNIXTIME="${RUN_TIME:-$(date +%s)}"
AGENT_CLASS="${AGENT_CLASS:-ppo}"
ENV_NAME="${ENV_NAME:-hopper}"
MAX_BUDGET="${MAX_BUDGET:-2500000}"

TD="${TD:-dtd}" # baseline / naive / dtd
NOISE_LVL="${NOISE_LVL:-0.01}"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

INCUMBENT_PATH="${SCRIPT_DIR}/incumbent/${AGENT_CLASS}/${ENV_NAME}/${TD}/noise_lvl${NOISE_LVL_STR}.sh"
if [[ ! -f "${INCUMBENT_PATH}" ]]; then
    echo "Missing incumbent parameter file: ${INCUMBENT_PATH}" >&2
    exit 1
fi

source "${INCUMBENT_PATH}"


echo "Running experiment with:"
echo "ENV_NAME: $ENV_NAME"
echo "NOISE_LVL: $NOISE_LVL"
echo "TD: $TD"
echo "RUN_TIME: $UNIXTIME"
echo "INCUMBENT_PATH: $INCUMBENT_PATH"
echo "=============================="


cd "${REPO_ROOT}" || exit 1

"${PYTHON_BIN}" -m dtd.${AGENT_CLASS}.main \
    hydra.run.dir="${REPO_ROOT}/dtd/configs/logs/${AGENT_CLASS}/${ENV_NAME}/${TD}/noise${NOISE_LVL_STR}/${UNIXTIME}" \
    env.name=${ENV_NAME} \
    env.noise_lvl=${NOISE_LVL} \
    algorithm=${AGENT_CLASS}_${TD} \
    algorithm.total_timesteps=${MAX_BUDGET} \
    algorithm.num_env_steps_per_update=${NUM_ENV_STEPS_PER_UPDATE} \
    algorithm.num_epochs_per_update=${NUM_EPOCHS_PER_UPDATE} \
    algorithm.minibatch_size=${MINIBATCH_SIZE} \
    algorithm.model_kwargs.learning_rate=${LEARNING_RATE} \
    algorithm.model_kwargs.mix_ratio=${MIX_RATIO} \
    algorithm.model_kwargs.gae_lambda=${GAE_LAMBDA} \
    algorithm.model_kwargs.clip_range=${CLIP_RANGE} \
    algorithm.model_kwargs.normalize_advantage=${NORMALIZE_ADVANTAGE} \
    algorithm.model_kwargs.vf_coef=${VF_COEF} \
    algorithm.model_kwargs.ent_coef=${ENT_COEF} \
    run_time=${UNIXTIME} \
    "$@"