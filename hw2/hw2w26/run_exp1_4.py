"""Run Experiment 1.4 from Part5_CNN_Experiments.ipynb

This script runs the ResNet configurations:
- K=[32] fixed with L=8,16,32
- K=[64, 128, 256] fixed with L=2,4,8

Usage (from workspace root):
    python run_exp1_4.py
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
lr = 1e-3
reg = 1e-3
model_type = 'resnet'

results = []

def run_configs():
    # --- Part 1: K=[32], L in [8, 16, 32] ---
    K1 = [32]
    for L in [8, 16, 32]:
        run_name = "exp1_4"
        print(f"\n=== Running {run_name} ===")
        
        cfg = dict(
            run_name=run_name,
            seed=seed,
            bs_train=bs_train,
            batches=batches,
            epochs=epochs,
            early_stopping=early_stopping,
            pool_every=pool_every,
            filters_per_layer=K1,
            layers_per_block=L,
            pool_every=pool_every,
            hidden_dims=hidden_dims,
            model_type=model_type,
            lr=lr,
            reg=reg,
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

    # --- Part 2: K=[64, 128, 256], L in [2, 4, 8] ---
    K2 = [64, 128, 256]
    for L in [2, 4, 8]:
        # P=L ensures one pool after each filter stage
        pool_every = L
        
        run_name = f"exp1_4_L{L}_K{'-'.join(map(str, K2))}"
        print(f"\n=== Running {run_name} ===")
        
        cfg = dict(
            run_name=run_name,
            seed=seed,
            bs_train=bs_train,
            batches=batches,
            epochs=epochs,
            early_stopping=early_stopping,
            filters_per_layer=K2,
            layers_per_block=L,
            pool_every=pool_every,
            hidden_dims=hidden_dims,
            model_type=model_type,
            lr=lr,
            reg=reg,
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