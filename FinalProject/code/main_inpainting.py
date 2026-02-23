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

# Set up pathes
image_num = 1
image_path = "../data/images/image0" + str(image_num) + ".jpg"
mask_path = "../data/masks/mask0" + str(image_num) + ".jpg"
saving_output_path = "../data/outputs/main_outputs/inpaint0" + str(image_num) + ".jpg"

# Set up pipeline components
pipe_comp_dict = utils.create_pipeline_components(model_id, device, dtype)
tokenizer = pipe_comp_dict["tokenizer"]
text_encoder = pipe_comp_dict["text_encoder"]
unet = pipe_comp_dict["unet"]
vae = pipe_comp_dict["vae"]
scheduler = pipe_comp_dict["scheduler"]

# Set up loss function for latent space
loss_fn = utils.LatentLoss()

prompt = "A dog sitting, on grass. shot on a phone camera. trees and mountains in the background."

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
img_tensor = utils.image_to_tensor(image_path=image_path, tensor_size=vae_sample_size, is_mask=False, device=device, dtype=dtype) # Image to tensor
with torch.no_grad():  # Transform image_tensor to latent space representation
    latent_original_image = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor

# Transform mask to latent space size binary torch tensor based on the masked region.
mask_latent_tensor = utils.image_to_tensor(image_path=mask_path, tensor_size=latent_sample_size, is_mask=True, device=device, dtype=dtype)

scheduler.set_timesteps(25)
guidance_scale = 7 # Control the CFG. Higher values -> higher dependency on the prompt. Values should be around 6-12.
optimization_scale = 0.1

# Applying the diffusion iterations.
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

    # Apply Classifier Free Guidance
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(
        noise_pred, t, latents
    ).prev_sample

    # Forward-noise the original latent to the current timestep t
    alpha_prod_t = scheduler.alphas_cumprod[t]
    noise = torch.randn_like(latents)
    latents_original_image_noised = (
        latent_original_image * alpha_prod_t.sqrt() +
        noise * (1 - alpha_prod_t).sqrt()
    )
    
    # Clamp known region. Note: '*' is elemnent-wise multiplication.
    latents = mask_latent_tensor * latents_original_image_noised + (1 - mask_latent_tensor) * latents

    # Apply latent space optimization with the loss function
    latents = latents.detach().requires_grad_(True)
    loss = loss_fn(diffusion_latent=latents, original_image_latent=latents_original_image_noised, mask=mask_latent_tensor)
    grad = torch.autograd.grad(loss, latents)[0]
    latents = latents - optimization_scale * grad
    latents = latents.detach()

latents = latents / vae.config.scaling_factor

with torch.no_grad():
    image = vae.decode(latents).sample

utils.tensor_to_image(image_tensor=image, saving_path=saving_output_path)
