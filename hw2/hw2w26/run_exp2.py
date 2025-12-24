"""Run Experiment 2 from Part5_CNN_Experiments.ipynb

This script runs the custom 'YourCNN' configurations:
- Comparing different widths K=[32, 64, 128] 
- Using the optimal depth L discovered in Exp 1 (typically L=2 or L=4)

Usage (from workspace root):
    python run_exp2.py
"""

import os
import time
from pprint import pprint

try:
    from hw2.experiments import cnn_experiment
except Exception as e:
    print("Failed to import hw2.experiments.cnn_experiment:", e)
    raise

# Common hyperparameters based on previous recommendations
seed = 42
bs_train = 128
batches = 400     # Ensures over 30,000 images (400 * 128 = 51,200)
epochs = 30
early_stopping = 5
pool_every = 3
hidden_dims = [256]
lr = 1e-3
reg = 1e-3
model_type = 'yours' # Ensure this matches your YourCNN mapping in MODEL_TYPES

results = []
run_name = "exp2" 

def run_configs():
    # --- Experiment 2: Testing YourCNN with varying widths ---
    # We use L=2 as the base because it was the most stable in Exp 1.1
    K =  [32, 64, 128]
    
    for L in [3, 6, 9, 12]:
        
        # Following your preferred clean naming convention for the legend

        print(f"\n=== Running {run_name} with L={L}, K={K} ===")
        
        cfg = dict(
            run_name=run_name,
            seed=seed,
            bs_train=bs_train,
            batches=batches,
            epochs=epochs,
            early_stopping=early_stopping,
            filters_per_layer=K,
            layers_per_block=L,
            pool_every=L,
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

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    run_configs()
    print('\nSummary of Experiment 2:')
    pprint(results)