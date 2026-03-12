import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from torchvision import transforms
import utils

# Set up device, dtype and model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
model_id = "sd2-community/stable-diffusion-2-base"

# Set up data pathes
image_num = 1
image_path, mask_path, saving_output_path = utils.get_paths(image_num=image_num, is_vanilla=True)
prompt = utils.get_prompt(image_num)

# Set up pipeline components
pipe = utils.create_pipeline_components(model_id, device, dtype)
tokenizer, text_encoder = pipe["tokenizer"], pipe["text_encoder"]
unet, vae, scheduler = pipe["unet"], pipe["vae"], pipe["scheduler"]

# Embedd the prompt. workflow: prompt -> tokens -> embedding
text_embeddings = utils.get_text_embeddings(prompt, tokenizer,text_encoder, device)
# Embedd an empty prompt for the Classifier Free Guidance
uncond_embeddings = utils.get_text_embeddings("", tokenizer,text_encoder, device)

# Sample random latent space data
batch_size = 1
latent_sample_size = unet.config.sample_size
latents = torch.randn(
    (
        batch_size,
        unet.config.in_channels,
        latent_sample_size,
        latent_sample_size,
    ),
    device=device,
    dtype=dtype,
) * scheduler.init_noise_sigma # Scale the noisy latent by the scheduler scalar, as it was trained on

# Transform image to torch tensor and then to latent space
vae_sample_size = vae.config.sample_size #  size of feature space
img_tensor = utils.image_to_tensor(image_path=image_path, tensor_size=vae_sample_size, device=device, dtype=dtype) # Image to tensor
with torch.no_grad():  # Transform image_tensor to latent space representation
    latent_original_image = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor

# Transform mask to latent space size binary torch tensor based on the masked region.
mask_latent_tensor = utils.mask_to_tensor(mask_path=mask_path, tensor_size=latent_sample_size, blur_radius=0, mask_shrink=0, device=device, dtype=dtype)

scheduler.set_timesteps(25)
guidance_scale = 7

for t in scheduler.timesteps:
    latent_model_input = scheduler.scale_model_input(latents, t) # Note: scale_model_input actually do nothing with current schedule, but safer. 

    with torch.no_grad():
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        noise_pred_uncond = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=uncond_embeddings
        ).sample

    # Apply CFG
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(
        noise_pred, t, latents
    ).prev_sample

    # Noise the original image
    noise = torch.randn_like(latents)
    latents_original_image_noised = scheduler.add_noise(
        latent_original_image,
        noise,
        t
    )
    
    # Clamp known regions
    latents = mask_latent_tensor * latents_original_image_noised + (1 - mask_latent_tensor) * latents

latents = latents / vae.config.scaling_factor

with torch.no_grad():
    image = vae.decode(latents).sample

image = utils.tensor_to_image(image_tensor=image)
image.save(saving_output_path)