import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import utils

# PSNR (Peak Signal-to-Noise Ratio): Quantitative measure of Background Preservation.
# Desired Result: HIGHER is better (30+ dB is the standard for high quality).
# 
# Interpretation: 
# Compares the unmasked 'known' pixels of the original image to the final output. 
# It proves the inpainting process didn't blur or shift the original background. 
# Our pixel-space alpha-blending specifically aims to maximize this value 
# compared to vanilla latent-clamping which often suffers from border bleeding.
def calculate_background_psnr(image_num, generated_path):
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Paths
    orig_path = f"../data/images/image0{image_num}.jpg"
    mask_path = f"../data/masks/mask0{image_num}.jpg"
    
    # 1. Load the generated image and convert to tensor [0, 1]
    gen = to_tensor(Image.open(generated_path).convert("RGB"))
    
    # 2. Use YOUR utils to load the original image and mask exactly as the pipeline did
    # Assuming the target size is the height of the generated image (e.g., 512)
    target_size = gen.shape[1] 
    
    # utils returns [1, C, H, W], usually in range [-1, 1] or [0, 1]. 
    # We will save it to a temporary PIL image and load it back to guarantee matching formats.
    orig_tensor = utils.image_to_tensor(orig_path, target_size, is_mask=False, device=device, dtype=dtype)
    mask_tensor = utils.image_to_tensor(mask_path, target_size, is_mask=True, device=device, dtype=dtype)
    
    # Save and reload to ensure identical color scaling [0, 1] as the generated image
    utils.tensor_to_image(orig_tensor, "temp_orig.jpg")
    orig = to_tensor(Image.open("temp_orig.jpg").convert("RGB"))
    
    # 3. Format the mask (1 for background, 0 for hole)
    mask = mask_tensor.squeeze(0).cpu()
    mask = (mask > 0.5).float() 
    
    # 4. Calculate Mean Squared Error (MSE) ONLY on the background
    mse = F.mse_loss(orig * mask, gen * mask, reduction='sum') / mask.sum()
    
    if mse == 0:
        return float('inf') # Perfect match
    
    # Calculate PSNR
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse.item())
    
    return psnr

if __name__ == "__main__":
    image_num = 1
    
    # Measure Baseline (No pixel blending)
    baseline_psnr = calculate_background_psnr(image_num, f"../data/outputs/vanilla_outputs/inpaint0{image_num}.jpg")
    print(f"Baseline Background PSNR: {baseline_psnr:.2f} dB")

    # Measure Improved (With pixel blending)
    improved_psnr = calculate_background_psnr(image_num, f"../data/outputs/main_outputs/inpaint0{image_num}.jpg")
    print(f"Improved Background PSNR: {improved_psnr:.2f} dB")
