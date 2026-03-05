import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import argparse
import utils

# 1. Setup
parser = argparse.ArgumentParser()
parser.add_argument("--image_num", type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
model_id = "sd2-community/stable-diffusion-2-base"

# 2. Loading
image_path = f"../data/images/image0{args.image_num}.jpg"
mask_path = f"../data/masks/mask0{args.image_num}.jpg"
saving_output_path = f"../data/outputs/main_outputs/inpaint0{args.image_num}.jpg"

with open("../data/prompts.txt", "r") as f:
    prompt = f.read().splitlines()[args.image_num-1]

pipe = utils.create_pipeline_components(model_id, device, dtype)
tokenizer, text_encoder = pipe["tokenizer"], pipe["text_encoder"]
unet, vae, scheduler = pipe["unet"], pipe["vae"], pipe["scheduler"]

text_embeddings = utils.get_text_embeddings(prompt, tokenizer, text_encoder, device)
uncond_embeddings = utils.get_text_embeddings("", tokenizer, text_encoder, device)

# 3. Latents and Tensors
latent_size = unet.config.sample_size
latents = torch.randn((1, 4, latent_size, latent_size), device=device, dtype=dtype) * scheduler.init_noise_sigma

img_size = vae.config.sample_size
img_tensor = utils.image_to_tensor(image_path, img_size, is_mask=False, device=device, dtype=dtype)
mask_pixel_tensor = utils.image_to_tensor(mask_path, img_size, is_mask=True, device=device, dtype=dtype)

with torch.no_grad():
    latent_original = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor

mask_latent = utils.image_to_tensor(mask_path, latent_size, is_mask=True, device=device, dtype=dtype)
soft_mask_latent = TF.gaussian_blur(mask_latent, kernel_size=5, sigma=0.7)

# 4. Smoothness Loss
def total_variation_loss(latent):
    return torch.pow(latent[:,:,1:,:] - latent[:,:,:-1,:], 2).mean() + \
           torch.pow(latent[:,:,:,1:] - latent[:,:,:,:-1], 2).mean()

# 5. Diffusion Loop
scheduler.set_timesteps(30)
num_warmup_steps = 25 # No loss for the first warmup steps
torch.manual_seed(0)
noise = torch.randn_like(latents)

for i, t in enumerate(scheduler.timesteps):
    latent_input = scheduler.scale_model_input(latents, t)
    
    with torch.no_grad():
        noise_pred_text = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond = unet(latent_input, t, encoder_hidden_states=uncond_embeddings).sample
        orig_noised = scheduler.add_noise(latent_original, noise, t)

    noise_pred = noise_pred_uncond + 3.0 * (noise_pred_text - noise_pred_uncond)

    # Apply Optimization ONLY after the warmup period
    if i > num_warmup_steps:
        latents = latents.detach().requires_grad_(True)
        latents_next = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Get Feature Loss
        with torch.no_grad():
            orig_features = unet(orig_noised, t, encoder_hidden_states=text_embeddings).sample
        gen_features = unet(latents_next, t, encoder_hidden_states=text_embeddings).sample
        
        loss = total_variation_loss(latents_next) + (0.3 * F.mse_loss(gen_features * mask_latent, orig_features * mask_latent))
        grad = torch.autograd.grad(loss, latents)[0]
        
        with torch.no_grad():
            latents = latents_next - 0.5 * grad
            latents = soft_mask_latent * orig_noised + (1 - soft_mask_latent) * latents
    else:
        # Vanilla Step during Warmup
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = mask_latent * orig_noised + (1 - mask_latent) * latents
    
    latents = latents.detach()

latents = latents / vae.config.scaling_factor

with torch.no_grad():
    image = vae.decode(latents).sample

utils.tensor_to_image(image_tensor=image, saving_path=saving_output_path)
