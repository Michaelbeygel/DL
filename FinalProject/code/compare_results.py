import sys
import gc
import os
import torch
import pandas as pd
from tqdm import tqdm
from unittest.mock import MagicMock
from PIL import Image, ImageChops
import torchvision.transforms as T

# 1. ENVIRONMENT & BUG FIXES
sys.modules["diffusers.models.attention_dispatch"] = MagicMock()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
import utils
from main_inpainting import MainPipeline
from vanilla_inpainting import VanillaPipeline

# --- LOCAL FUNCTIONS ---

def generate_mask(source, target):
    """Generates a binary mask by comparing source and target images."""
    src_rgb = source.convert("RGB")
    tar_rgb = target.convert("RGB")
    diff = ImageChops.difference(src_rgb, tar_rgb)
    return diff.convert("L").point(lambda x: 255 if x > 15 else 0)

def image_to_tensor_local(pil_image, res=512, device="cuda", dtype=torch.float16):
    """Converts a PIL image to a normalized tensor [-1, 1]."""
    transform = T.Compose([
        T.Resize((res, res), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_t = transform(pil_image.convert("RGB")).unsqueeze(0)
    return img_t.to(device, dtype=dtype)

# --- BENCHMARK LOGIC ---

def run_integrated_benchmark(num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    print("Initializing Pipelines...")
    main_pipe = MainPipeline()
    vanilla_pipe = VanillaPipeline()
    
    eval_m = utils.InpaintingEvaluator(device, dtype)
    eval_v = utils.InpaintingEvaluator(device, dtype)
    
    print("Connecting to PIPE dataset...")
    
    dataset = load_dataset(
        "paint-by-inpaint/PIPE", 
        split="test", 
        streaming=True
    )
    
    results = []

    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Benchmark")):
        if i >= num_samples: break
        
        try:
            if sample is None: continue

            source = sample.get("source_img")
            target = sample.get("target_img")
            prompt = sample.get("Instruction_VLM-LLM")

            mask = generate_mask(source, target)

            if any(val is None for val in [source, target, mask, prompt]):
                continue

            # Visual Debug: Grid of Source | Mask | Target (separate panels)
            if i < 5:
                print(f"\n[DEBUG] Sample {i} Prompt: {prompt}")
                # Create a canvas for three 512x512 images side-by-side
                debug_grid = Image.new('RGB', (1536, 512))
                
                # Panel 1: Source (The image with the object missing)
                debug_grid.paste(source.resize((512, 512)), (0, 0))
                
                # Panel 2: Mask (The white footprint of the change)
                # Convert to RGB so it can be pasted into the RGB canvas
                debug_grid.paste(mask.convert("RGB").resize((512, 512)), (512, 0))
                
                # Panel 3: Target (The original ground truth image)
                debug_grid.paste(target.resize((512, 512)), (1024, 0))
                
                debug_grid.save(f"sample_{i}_check.png")
                print(f"[DEBUG] Saved 'sample_{i}_check.png' (Source | Mask | Target)")

            # 2. RUN MODELS
            m_out = main_pipe.run_main(source, mask, prompt)
            v_out = vanilla_pipe.run_vanilla(source, mask, prompt)
            
            # --- CRITICAL FIX: SAFETY WRAPPER ---
            # If the pipeline returns a PIL Image, convert it to a [-1, 1] tensor
            # so utils.evaluate_step (line 360) doesn't crash on division.
            if isinstance(m_out, Image.Image):
                m_tensor = image_to_tensor_local(m_out, 512, device, dtype)
            else:
                m_tensor = m_out

            if isinstance(v_out, Image.Image):
                v_tensor = image_to_tensor_local(v_out, 512, device, dtype)
            else:
                v_tensor = v_out
            # ------------------------------------

            # 3. TARGET CONVERSION
            tar_tensor = image_to_tensor_local(target, 512, device, dtype)

            # 4. EVALUATION
            res_m = eval_m.evaluate_step(m_tensor, tar_tensor, prompt, is_main=True)
            res_v = eval_v.evaluate_step(v_tensor, tar_tensor, prompt, is_main=True)

            results.append({
                "Sample": i, 
                "M_PSNR": res_m["PSNR"], "V_PSNR": res_v["PSNR"],
                "M_SSIM": res_m["SSIM"], "V_SSIM": res_v["SSIM"],
                "M_CLIP": res_m["CLIP"], "V_CLIP": res_v["CLIP"]
            })
            
        except Exception as e:
            print(f"\n[!] Error on sample {i}: {e}")
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # Final reporting
    df = pd.DataFrame(results)
    if not df.empty:
        final_fid_m = eval_m.compute_final_fid()
        final_fid_v = eval_v.compute_final_fid()
        
        print("\n" + "="*45)
        print("      TECHNION PROJECT: BENCHMARK SUMMARY")
        print("="*45)
        print(df.mean(numeric_only=True))
        print(f"Main Model FID:    {final_fid_m:.4f}")
        print(f"Vanilla Model FID: {final_fid_v:.4f}")
        print("="*45)
        
        df.to_csv("benchmark_results.csv", index=False)
    else:
        print("\n[!] No samples processed successfully.")

if __name__ == "__main__":
    run_integrated_benchmark(750)