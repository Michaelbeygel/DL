import sys
from unittest.mock import MagicMock

# 1. Bypass the Flash Attention registration error
sys.modules["diffusers.models.attention_dispatch"] = MagicMock()

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import utils
import os

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16  # Use float16 for UNet to save VRAM
model_id = "sd2-community/stable-diffusion-2-base"
image_num = 1
zeta_prime = 0.1      # Hyperparameter for step size
guidance_scale = 7.5
num_inference_steps = 200

# 2. Load Components
pipe = utils.create_pipeline_components(model_id, device, dtype)
tokenizer, text_encoder, unet, vae, scheduler = (
    pipe["tokenizer"], pipe["text_encoder"], 
    pipe["unet"], pipe["vae"], pipe["scheduler"]
)

vae_size = vae.config.sample_size 

# 3. Load Data
# y = A(x0) + n
y_pixel = utils.image_to_tensor(f"../data/images/image{image_num:02d}.jpg", vae_size, device, dtype)
# The mask: 1.0 for unmasked pixels (keep), 0.0 for pixels to inpaint
mask = utils.mask_to_tensor(f"../data/masks/mask{image_num:02d}.jpg", vae_size, 0, 0, device, dtype)

prompt = utils.get_prompt(image_num)
text_embeddings = utils.get_text_embeddings(prompt, tokenizer, text_encoder, device)
uncond_embeddings = utils.get_text_embeddings("", tokenizer, text_encoder, device)
context = torch.cat([uncond_embeddings, text_embeddings])

# 4. Initialize Latents
scheduler.set_timesteps(num_inference_steps)
# x_N ~ N(0, I) 
latents = torch.randn((1, unet.config.in_channels, 64, 64), device=device, dtype=dtype)
latents = latents * scheduler.init_noise_sigma

# --- Algorithm 1: Diffusion Posterior Sampling (DPS) --- 
for i, t in enumerate(scheduler.timesteps):
    # Prepare input for classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # A. Score Estimation (UNet Forward)
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=context).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # B. The Likelihood Approximation (The Nudge)
    # DPS requires gradients with respect to the current latents
    latents = latents.detach().requires_grad_(True)
    
    # Predict x_hat_0 using Tweedie's formula
    # In 'diffusers', scheduler.step calculates the predicted original sample (x_hat_0)
    scheduler_output = scheduler.step(noise_pred, t, latents)
    x_hat_0 = scheduler_output.pred_original_sample

    # Compute gradient of the likelihood approximation p(y|x_hat_0)
    with torch.enable_grad():
        # Forward Operator A(.) is the VAE decoding
        # We decode in float32 for numerical stability in loss calculation
        vae.to(torch.float32)
        decoded_x0 = vae.decode(x_hat_0.to(torch.float32) / vae.config.scaling_factor).sample
        
        # Calculate loss ONLY on unmasked parts 
        # ||y - A(x_hat_0)||^2
        residual = mask.to(torch.float32) * (y_pixel.to(torch.float32) - decoded_x0)
        loss = torch.norm(residual)**2
        
        # Backpropagate to latents
        grad = torch.autograd.grad(loss, latents)[0]
        vae.to(dtype) # Return VAE to half precision for memory

    with torch.no_grad():
        # Adaptive step size: zeta_i = zeta' / ||y - A(x_hat_0)|| 
        norm_residual = torch.norm(residual) + 1e-6
        zeta_i = zeta_prime / norm_residual
        
        # Update latents: x_i = x_i - zeta_i * grad
        latents = latents - zeta_i * grad

    # C. Ancestral Sampling Move
    # Move from x_i to x_{i-1} using the nudged latents
    with torch.no_grad():
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    latents = latents.detach()

    if i % 10 == 0:
        print(f"Step {i}/{num_inference_steps} | Loss: {loss.item():.4f}")

# 5. Final Decode and Save
with torch.no_grad():
    image_tensor = vae.decode(latents / vae.config.scaling_factor).sample
    final_image = utils.tensor_to_image(image_tensor)
    
    output_dir = "../data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/multi_step_dps_image{image_num:02d}.jpg"
    final_image.save(save_path)
    print(f"Success! Result saved to {save_path}")
