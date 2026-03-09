import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import utils
from main_inpainting import run_main # change to the the real functiuon name
from vanilla_inpainting import run_vanilla # change to the the real functiuon name

def run_benchmark(num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = utils.InpaintingEvaluator(device)
    
    # Load PIPE dataset (No git clone needed!)
    dataset = load_dataset("paint-by-inpaint/PIPE", split="test")
    results = []

    for i in tqdm(range(num_samples), desc="Comparing Main vs Vanilla"):
        sample = dataset[i]
        
        # Extract inputs for your functions
        prompt = sample["Instruction_VLM-LLM"]
        source_img = sample["source_img"] # Image with object removed
        mask_img = sample["mask"]         # The binary mask
        target_img = sample["target_img"] # The "Answer Key" (Ground Truth)

        # Run your logic (Ensure they return: tensor, pil_image)
        # tensor: (1, 3, 512, 512) normalized [-1, 1]
        v_tensor = run_vanilla(source_img, mask_img, prompt)
        m_tensor = run_main(source_img, mask_img, prompt)
        
        # Prepare Ground Truth for comparison
        target_tensor = utils.image_to_tensor(target_img, 512, device, torch.float16)

        # Calculate Metrics
        m_v = evaluator.evaluate(v_tensor, target_tensor)
        m_m = evaluator.evaluate(m_tensor, target_tensor)

        results.append({
            "Sample": i,
            "Vanilla_PSNR": m_v["PSNR"], "Main_PSNR": m_m["PSNR"],
            "Vanilla_LPIPS": m_v["LPIPS"], "Main_LPIPS": m_m["LPIPS"]
        })

    # Output a summary table
    df = pd.DataFrame(results)
    print("\n--- Mean Performance ---")
    print(df.mean(numeric_only=True))
    df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    run_benchmark(10)