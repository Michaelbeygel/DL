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

def generate_mask(source, target):
    diff = ImageChops.difference(source.convert("RGB"), target.convert("RGB"))
    return diff.convert("L").point(lambda x: 255 if x > 15 else 0)

def run_benchmark(num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = utils.InpaintingEvaluator(device)
    dataset = load_dataset("paint-by-inpaint/PIPE", split="test", streaming=True)
    results = []

    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Benchmark")):
        if i >= num_samples: break
        
        source, target = sample["source_img"], sample["target_img"]
        mask = generate_mask(source, target)
        prompt = sample["Instruction_VLM-LLM"]
        
        m_tensor = run_main(source, mask, prompt)
        v_tensor = run_vanilla(source, mask, prompt)
        tar_tensor = utils.image_to_tensor_from_pil(target, 512, device, torch.float16)

        # Evaluation including CLIP Score
        res_v = evaluator.evaluate(v_tensor, tar_tensor, prompt)
        res_m = evaluator.evaluate(m_tensor, tar_tensor, prompt)

        results.append({
            "Sample": i, 
            "Vanilla_PSNR": res_v["PSNR"], "Main_PSNR": res_m["PSNR"], 
            "Vanilla_CLIP": res_v["CLIP"], "Main_CLIP": res_m["CLIP"]
        })
        torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(results)
    if not df.empty:
        print("\n--- Mean Results ---")
        print(df.mean(numeric_only=True))
        df.to_csv("benchmark_results.csv", index=False)
        print("Results saved to benchmark_results.csv")
    else:
        print("No samples were processed.")

if __name__ == "__main__":
    run_benchmark(1)