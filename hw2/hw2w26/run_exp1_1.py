"""Run Experiment 1.1 from Part5_CNN_Experiments.ipynb

This script runs the configurations described in the notebook:
- K=32 with L in [2,4,8,16]
- K=64 with L in [2,4,8,16]

It calls `hw2.experiments.cnn_experiment` for each configuration and
saves results in the `results/` folder (as the notebook expects).

Usage (from workspace root):
    python run_exp1_1.py

Note: Training can be long; these are the configurations to run. Adjust
`batches`/`epochs`/`bs_train` as needed for quicker tests.
"""

import os
import time
from pprint import pprint

try:
    from hw2.experiments import cnn_experiment
except Exception as e:
    print("Failed to import hw2.experiments.cnn_experiment:", e)
    raise

# Common hyperparameters (tunable)
seed = 42
bs_train = 128
batches = 500
epochs = 50
early_stopping = 5
pool_every = 3
hidden_dims = [512]
lr = 1e-3  # learning rate
reg = 1e-3  # regularization
model_type = 'cnn'  # as in notebook

results = []

def run_configs():
    for K in ([32], [64]):
        k_val = K[0]
        for L in [2, 4, 8, 16]:
            run_name = "exp1_1"
            print(f"\n=== Running {run_name} ===")
            cfg = dict(
                run_name=run_name,
                seed=seed,
                bs_train=bs_train,
                batches=batches,
                epochs=epochs,
                early_stopping=early_stopping,
                filters_per_layer=K,
                layers_per_block=L,
                pool_every=pool_every,
                hidden_dims=hidden_dims,
                lr=lr,
                reg=reg,
                model_type=model_type,
            )
            start = time.time()
            try:
                cnn_experiment(**cfg)
                status = 'ok'
            except Exception as e:
                print(f"Run {run_name} failed: {e}")
                status = f'error: {e}'
            duration = time.time() - start
            results.append((run_name, status, duration))
            print(f"Finished {run_name}: {status} ({duration:.1f}s)")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    run_configs()
    print('\nSummary:')
    pprint(results)
    print('\nNote: adjust `batches`/`epochs`/`bs_train` in this script for faster testing.')