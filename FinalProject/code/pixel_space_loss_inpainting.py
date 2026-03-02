import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os
import utils

# 1. Setup Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
model_id = "sd2-community/stable-diffusion-2-base"

# Paths
image_num = 1
image_path = f"../data/images/image0{image_num}.jpg"
mask_path = f"../data/masks/mask0{image_num}.jpg"
saving_output_path = f"../data/outputs/dps_stable_multistep.jpg"
os.makedirs("../data/outputs/", exist_ok=True)

# Load Components
pipe_comp_dict = utils.create_pipeline_components(model_id, device, dtype)
tokenizer, text_encoder, unet, vae, scheduler = (
    pipe_comp_dict["tokenizer"], pipe_comp_dict["text_encoder"], 
    pipe_comp_dict["unet"], pipe_comp_dict["vae"], pipe_comp_dict["scheduler"]
)

# Freeze networks
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# 2. Pre-processing
with open("../data/prompts.txt", "r") as f:
    prompt = f.read().splitlines()[image_num-1]

text_embeddings = utils.get_text_embeddings(prompt, tokenizer, text_encoder, device)
uncond_embeddings = utils.get_text_embeddings("", tokenizer, text_encoder, device)
text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])

vae_size = vae.config.sample_size
y = utils.image_to_tensor(image_path, vae_size, False, device, dtype)
mask = utils.image_to_tensor(mask_path, vae_size, True, device, dtype).unsqueeze(0)

# 3. Algorithm Parameters
scheduler.set_timesteps(999)
zeta_prime = 0.5
guidance_scale = 5.0
num_grad_steps = 3  # Internal optimization steps per diffusion step

x_i = torch.randn((1, 4, 64, 64), device=device, dtype=dtype) * scheduler.init_noise_sigma
loss_history, timestep_history, intermediate_images = [], [], []



# --- Algorithm 1: DPS - Gaussian Diffusion Loop ---
for step_idx, t in enumerate(scheduler.timesteps):
    x_i = x_i.detach().requires_grad_(True)
    
    # UNet Forward Pass (Once per diffusion step)
    latent_model_input = torch.cat([x_i] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    with torch.no_grad():
        noise_pred_cfg = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_cfg).sample
        noise_pred_uncond, noise_pred_text = noise_pred_cfg.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # --- STABILIZED INTERNAL GRADIENT ITERATION ---
    for g_step in range(num_grad_steps):
        x_i = x_i.detach().requires_grad_(True)
        
        with torch.enable_grad():
            alpha_bar_i = scheduler.alphas_cumprod[t].to(dtype=dtype)
            x_hat_0 = (x_i - (1 - alpha_bar_i).sqrt() * noise_pred) / alpha_bar_i.sqrt()

            # STABILITY FIX: Force decode in float32 to prevent NaN
            decoded_x0 = vae.to(torch.float32).decode(x_hat_0.to(torch.float32) / vae.config.scaling_factor).sample
            
            # Loss in float32
            difference = mask.to(torch.float32) * (y.to(torch.float32) - decoded_x0)
            loss = torch.sum(difference**2)

        grad = torch.autograd.grad(loss, x_i)[0]
        
        # STABILITY FIX: NaN/Inf Safety Valve
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            # NEW: Directional Normalization to prevent explosions
            grad_norm = torch.norm(grad)
            latent_norm = torch.norm(x_i)
            
            if grad_norm > 0:
                # Move proportional to latent magnitude, capped at 4% per sub-step
                max_shift = 0.04 / num_grad_steps
                step_size = (zeta_prime / num_grad_steps) * (latent_norm / (grad_norm + 1e-6))
                step_size = min(step_size, max_shift)
                
                x_i = x_i - (step_size * grad)

    # --- ANCESTRAL SAMPLING MOVE TO t-1 ---
    with torch.no_grad():
        prev_t = scheduler.timesteps[step_idx + 1] if step_idx < len(scheduler.timesteps) - 1 else -1
        alpha_bar_prev = scheduler.alphas_cumprod[prev_t].to(dtype=dtype) if prev_t >= 0 else torch.tensor(1.0, device=device, dtype=dtype)
        
        beta_i = 1 - (alpha_bar_i / alpha_bar_prev)
        sigma_tilde_i = ((1 - alpha_bar_prev) / (1 - alpha_bar_i) * beta_i).sqrt()
        z = torch.randn_like(x_i) if step_idx < len(scheduler.timesteps) - 1 else torch.zeros_like(x_i)

        coeff_xi = (alpha_bar_i / alpha_bar_prev).sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_i)
        coeff_x0 = (alpha_bar_prev).sqrt() * beta_i / (1 - alpha_bar_i)
        
        x_i = coeff_xi * x_i.detach() + coeff_x0 * x_hat_0.detach() + sigma_tilde_i * z

    # Tracking
    loss_history.append(loss.item())
    timestep_history.append(t.item())
    if step_idx % 50 == 0 or step_idx == len(scheduler.timesteps) - 1:
        # Calculate ratio for logging
        current_ratio = (torch.norm(step_size * grad) / torch.norm(x_i) * 100) if grad_norm > 0 else 0
        print(f"[t={t.item()}] Loss: {loss.item():.2f} | Guidance: {current_ratio:.2f}%")
        intermediate_images.append(decoded_x0.detach().cpu())
    
    vae.to(dtype)

# Final Save
vae.to(dtype)
with torch.no_grad():
    image = vae.decode(x_i / vae.config.scaling_factor).sample
utils.tensor_to_image(image, saving_output_path)

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(timestep_history, loss_history, marker='o', color='b')
plt.gca().invert_xaxis()
plt.title("Algorithm 1: Stable Multi-Step DPS")
plt.savefig("../data/outputs/dps_stable_loss.png"); plt.close()

grid_tensor = (torch.cat(intermediate_images, dim=0) / 2 + 0.5).clamp(0, 1)
save_image(make_grid(grid_tensor, nrow=4), "../data/outputs/dps_stable_grid.jpg")

print(f"Success. Check Guidance Ratio: should be controlled (max 4% per diffusion step).")