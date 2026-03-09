"""
Final Project: DPS Algorithm 1 (Gaussian) 
Implementation as per Chung et al. (2023)
"""

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import utils

# 1. Setup Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
model_id = "sd2-community/stable-diffusion-2-base"

# 2. Components & Loading (Page 3, Forward Model) [cite: 80, 82]
pipe = utils.create_pipeline_components(model_id, device, dtype)
tokenizer, text_encoder, unet, vae, scheduler = (
    pipe["tokenizer"], pipe["text_encoder"], 
    pipe["unet"], pipe["vae"], pipe["scheduler"]
)

image_num = 1
vae_size = vae.config.sample_size 

# y = A(x0) + n
y = utils.image_to_tensor(f"../data/images/image0{image_num}.jpg", vae_size, False, device, dtype)
mask = utils.image_to_tensor(f"../data/masks/mask0{image_num}.jpg", vae_size, True, device, dtype)

with open("../data/prompts.txt", "r") as f:
    prompt = f.read().splitlines()[image_num-1]
text_embeddings = utils.get_text_embeddings(prompt, tokenizer, text_encoder, device)
uncond_embeddings = utils.get_text_embeddings("", tokenizer, text_encoder, device)

# 3. Parameters (Algorithm 1)
scheduler.set_timesteps(100) 
num_grad_steps = 10  # Internal optimization steps per diffusion step
zeta_prime = 0.5
guidance_scale = 7.5

x_i = torch.randn((1, 4, 64, 64), device=device, dtype=dtype) * scheduler.init_noise_sigma

# --- Algorithm 1: Multi-Step DPS Loop ---
for i, t in enumerate(scheduler.timesteps):
    
    # 1. UNet Forward Pass (Once per diffusion step to save VRAM) 
    latent_input = scheduler.scale_model_input(x_i, t)
    with torch.no_grad():
        noise_pred_uncond = unet(latent_input, t, encoder_hidden_states=uncond_embeddings).sample
        noise_pred_text = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # 2. INTERNAL GRADIENT ITERATION (The Nudge) 
    for g_step in range(num_grad_steps):
        x_i = x_i.detach().requires_grad_(True)
        
        # A. Predict x_hat_0 [cite: 116, 136]
        output = scheduler.step(noise_pred, t, x_i, return_dict=True)
        x_hat_0 = output.pred_original_sample
        
        # B. Forward Operator A(.) in float32 
        with torch.enable_grad():
            decoded_x0 = vae.to(torch.float32).decode(x_hat_0.to(torch.float32) / vae.config.scaling_factor).sample
            
            # C. Masked Likelihood Loss [cite: 136, 164]
            diff = mask.to(torch.float32) * (y.to(torch.float32) - decoded_x0)
            loss = torch.norm(diff)**2
            
            # D. Backprop to Latent [cite: 162, 136]
            grad = torch.autograd.grad(loss, x_i)[0]
            
        with torch.no_grad():
            # Adaptive step size zeta_i [cite: 220]
            zeta_i = zeta_prime / (torch.norm(diff) + 1e-6)
            
            # Internal Nudge: Nudging the current x_i before the final sampling move
            x_i = x_i - zeta_i * grad

    # 3. ANCESTRAL SAMPLING MOVE [cite: 136, 211]
    # After refining x_i with grad steps, move to the next timestep t-1
    with torch.no_grad():
        x_i = scheduler.step(noise_pred, t, x_i).prev_sample
        
    x_i = x_i.detach()
    vae.to(dtype)
    
    if i % 20 == 0:
        print(f"[Step {i}] Likelihood Loss: {loss.item():.4f}")

# Final return 
with torch.no_grad():
    image = vae.decode(x_i / vae.config.scaling_factor).sample
utils.tensor_to_image(image, f"../data/outputs/multi_step_dps_image0{image_num}.jpg")
