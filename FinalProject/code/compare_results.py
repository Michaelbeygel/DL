import sys, gc, torch
from unittest.mock import MagicMock
sys.modules["diffusers.models.attention_dispatch"] = MagicMock()
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from PIL import ImageChops
import utils
from main_inpainting import run_main
from vanilla_inpainting import run_vanilla

def run_benchmark(num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    # We need two evaluators to track separate FID distributions
    eval_m = utils.InpaintingEvaluator(device, dtype)
    eval_v = utils.InpaintingEvaluator(device, dtype)
    
    dataset = load_dataset("paint-by-inpaint/PIPE", split="test", streaming=True)
    results = []

    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Benchmark")):
        if i >= num_samples: break
        
        source, target = sample["source_img"], sample["target_img"]
        mask = sample["mask"] # Use PIPE's built-in mask
        prompt = sample["Instruction_VLM-LLM"]
        
        # 1. Run Approaches
        m_tensor = run_main(source, mask, prompt)
        v_tensor = run_vanilla(source, mask, prompt)
        tar_tensor = utils.image_to_tensor_from_pil(target, 512, device, dtype)

        # 2. Evaluation
        # evaluate_step updates internal FID state and returns per-image metrics
        res_m = eval_m.evaluate_step(m_tensor, tar_tensor, prompt, is_main=True)
        res_v = eval_v.evaluate_step(v_tensor, tar_tensor, prompt, is_main=True)

        results.append({
            "Sample": i, 
            "M_PSNR": res_m["PSNR"], "V_PSNR": res_v["PSNR"],
            "M_SSIM": res_m["SSIM"], "V_SSIM": res_v["SSIM"],
            "M_CLIP": res_m["CLIP"], "V_CLIP": res_v["CLIP"]
        })
        
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Final Calculations
    df = pd.DataFrame(results)
    if not df.empty:
        # Calculate final dataset-wide FID
        final_fid_m = eval_m.compute_final_fid()
        final_fid_v = eval_v.compute_final_fid()
        
        print("\n--- Final Summary Results ---")
        means = df.mean(numeric_only=True)
        print(means)
        print(f"Main Model FID: {final_fid_m:.4f}")
        print(f"Vanilla Model FID: {final_fid_v:.4f}")
        
        # Save results
        df.to_csv("benchmark_results.csv", index=False)
        with open("fid_results.txt", "w") as f:
            f.write(f"Main FID: {final_fid_m}\nVanilla FID: {final_fid_v}")
    else:
        print("No samples were processed.")

if __name__ == "__main__":
    run_benchmark(1)