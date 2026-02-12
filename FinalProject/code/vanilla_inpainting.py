## Problems worth dealing with in current version:
# 1) When transforming the latent instance back to feature space, we lose the specific values of the given image!
#    That is because we are applying the given image noisy versions in the stable diffusion process in the latent space. 

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from torchvision import transforms
import utils

# Set up device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# Set up pathes
model_id = "sd2-community/stable-diffusion-2-base"
image_path = "../images/image01.jpg"
mask_path = "../images/mask01.jpg"

# Set up pipeline components
pipe_comp_dict = utils.create_pipeline_components(model_id, device, dtype)
tokenizer = pipe_comp_dict["tokenizer"]
text_encoder = pipe_comp_dict["text_encoder"]
unet = pipe_comp_dict["unet"]
vae = pipe_comp_dict["vae"]
scheduler = pipe_comp_dict["scheduler"]

prompt = "Portrait of a beautiful woman sitting at a Parisian cafe table, white sweater, holding a coffee cup, soft sunlight, blurred street background, high resolution, highly detailed skin, 8k, cinematic lighting."

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
        latent_sample_size, # Height = latent sample size
        latent_sample_size, # Widrh = latent sample size
    ),
    device=device,
    dtype=dtype,
) * scheduler.init_noise_sigma # Scale the noisy latent by the scheduler scalar, as it was trained on

# Transform image to torch tensor and then to latent space
vae_sample_size = vae.config.sample_size #  size of feature space
img_tensor = utils.image_to_tensor(image_path=image_path, tensor_size=vae_sample_size, device=device, dtype=dtype) # Image to tensor
with torch.no_grad():  # Transform image_tensor to latent space representation
    latent_space_image = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor

# Transform mask to latent space size torch tensor. Not using autoencoder.
mask_latent_tensor = utils.image_to_tensor(image_path=mask_path, tensor_size=latent_sample_size, device=device, dtype=dtype)

# 3) Each step apply multiplication between the mask and the diffusion step

scheduler.set_timesteps(50)
guidance_scale = 5 # Control the CFG. Higher values -> higher dependency on the prompt, more noise

# Applying the diffusion iterations.
for t in scheduler.timesteps:
    # scale_model_input actually do nothing with current schedule, but is safer to use it anyway.
    latent_model_input = scheduler.scale_model_input(latents, t) 

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

    # Apply Classifier Free Guidance
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(
        noise_pred, t, latents
    ).prev_sample

latents = latents / vae.config.scaling_factor

with torch.no_grad():
    image = vae.decode(latents).sample

utils.tensor_to_image(image_tensor=image, saving_path="generated.png")
