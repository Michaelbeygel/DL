#!/bin/bash

# SBATCH wrapper to run all Experiment 1.4 configurations sequentially
# Usage: sbatch sbatch_run_exp1_4.sh

# -- SLURM job settings (tweak as needed) --
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="exp1_4_all"
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
HIDDEN_DIMS=(100)
MODEL_TYPE=resnet
OUT_DIR=results
LR=0.001
REG=0.001

# Ensure results dir exists
mkdir -p "$OUT_DIR"

# --- Part A: Fixed K=[32], Varying L {8, 16, 32} ---
K="32"
for L in 8 16 32; do
    # Adjust P based on L to prevent spatial dimension from hitting 0
    if [ $L -eq 32 ]; then P=16; elif [ $L -eq 16 ]; then P=8; else P=4; fi
    
    RUN_NAME="exp1_4_L${L}_K${K}"
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
      -P $P \
      -H ${HIDDEN_DIMS[@]} \
      --lr $LR \
      --reg $REG \
      -M $MODEL_TYPE

    echo "--- Finished ${RUN_NAME} ---"
done

# --- Part B: Fixed K=[64, 128, 256], Varying L {2, 4, 8} ---
K="64 128 256"
for L in 2 4 8; do
    # P=L ensures pooling happens once after each of the 3 filter blocks
    P=$L
    
    RUN_NAME="exp1_4_L${L}_K64-128-256"
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
      -P $P \
      -H ${HIDDEN_DIMS[@]} \
      --lr $LR \
      --reg $REG \
      -M $MODEL_TYPE

    echo "--- Finished ${RUN_NAME} ---"
done

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

echo "Created sbatch_run_exp1_4.sh. Submit with: sbatch sbatch_run_exp1_4.sh"