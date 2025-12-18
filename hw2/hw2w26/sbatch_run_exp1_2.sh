#!/bin/bash

# SBATCH wrapper to run all Experiment 1.2 configurations sequentially
# Usage: sbatch sbatch_run_exp1_2.sh

# -- SLURM job settings (tweak as needed) --
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="exp1_2_all"
MAIL_USER=""
MAIL_TYPE=NONE

# Conda env settings
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

sbatch -N $NUM_NODES -c $NUM_CORES --gres=gpu:$NUM_GPUS --job-name $JOB_NAME --mail-user $MAIL_USER --mail-type $MAIL_TYPE -o "slurm-%N-%j.out" <<'EOF'
#!/bin/bash
set -euo pipefail

echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Activate conda
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Move to workspace root (assumes this script is submitted from workspace folder)
cd "$SLURM_SUBMIT_DIR"

# Common training args (tweak for faster testing)
BS_TRAIN=128
BATCHES=100
EPOCHS=50
EARLY_STOP=5
POOL_EVERY=2
HIDDEN_DIMS=(100)
LR=0.001  # learning rate
REG=0.001  # regularization
MODEL_TYPE=cnn
OUT_DIR=results

# Ensure results dir exists
mkdir -p "$OUT_DIR"

# Run configs: L in {2,4,8}, K in {32,64,128}
for L in 2 4 8; do
  for K in 32 64 128; do
    RUN_NAME="exp1_2_L${L}_K${K}"
    echo "--- Running ${RUN_NAME} ---"

    python -m hw2.experiments run-exp \
      -n "$RUN_NAME" \
      -o "$OUT_DIR" \
      -s 42 \
      --bs-train $BS_TRAIN \
      --batches $BATCHES \
      --epochs $EPOCHS \
      --early-stopping $EARLY_STOP \
      -K $K \
      -L $L \
      -P $POOL_EVERY \
      -H ${HIDDEN_DIMS[@]} \
      --lr $LR \
      --reg $REG \
      -M $MODEL_TYPE

    echo "--- Finished ${RUN_NAME} ---"
  done
done

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

echo "Created sbatch_run_exp1_2.sh. Submit with: sbatch sbatch_run_exp1_2.sh"}json```}043View.openapi_response_guidance_SANITIZE_OUTPUT_LIBRARY_JSON_JSON_INVALID_EXCEPTIONABILITY_UTF8_HEADERS_JSON_XMLIFIED_RESPONSE_LIMITATION_TRANSLATION_ERRORINVALID_JSON_ERROR_DEBUG_BLOCK_EXIT_CHAIN_SECURITY_POLICY_NOTICE_TEAM_OVERRIDE_AFTER_CALL_OUTPUT_POLICY_APOLOGY_TOO_LONG_FIXED_FORMAT_OUTPUT_ERROR_MODULE_COMPLETION_TOO_LONG_JSON_PARSE_ERROR_BAR_EXPLANATION_CONTINUE_EMBEDDED_MESSAGE_COMPLETE_CREATING_FILE_SUCCESS