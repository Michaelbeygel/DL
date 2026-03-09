import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
import utils
import torch.nn.functional as F

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run inpainting with specified image number")
parser.add_argument("--image_num", type=int, default=1, help="Image number to process (default: 1)")
args = parser.parse_args()
image_num = args.image_num

# Set up device, dtype and model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
model_id = "sd2-community/stable-diffusion-2-base"

# Set up data pathes
image_path, mask_path, saving_output_path = utils.get_paths(image_num=image_num, is_vanilla=False)

# Get prompt number 'image_num' from "prompts.txt" file.  
prompt = utils.get_prompt(image_num)

# Set up pipeline components
pipe = utils.create_pipeline_components(model_id, device, dtype)
tokenizer, text_encoder = pipe["tokenizer"], pipe["text_encoder"]
unet, vae, scheduler = pipe["unet"], pipe["vae"], pipe["scheduler"]

# Set up loss function for latent space
loss_fn = utils.LatentLoss()

# Embedd the prompt. workflow: prompt -> tokens -> embedding
text_embeddings = utils.get_text_embeddings(prompt, tokenizer,text_encoder, device)
# Embedd an empty prompt for the Classifier Free Guidance
uncond_embeddings = utils.get_text_embeddings("", tokenizer,text_encoder, device)

# Create a list of size 'num_of_inpaintings' of images. 
images = []
num_of_inpaintings = 5

# Set up a mask scheduler. This will help us change masks throughout the iterations of the pipeline.
timestemps = 25
latent_sample_size = unet.config.sample_size
mask_scheduler = utils.MaskScheduler(
    timestemps=timestemps, 
    mask_path=mask_path, 
    tensor_size=latent_sample_size,
    device=device,
    dtype=dtype
)

scheduler.set_timesteps(timestemps)
guidance_scale = 7 # Control the CFG. Higher values -> higher dependency on the prompt. Values should be around 6-12.

for inpaint_iteration in range(num_of_inpaintings):
    # Sample random latent space data
    batch_size = 1
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

    # Transform image:
    # Image -> torch tensor with filled mask areas -> latent space
    vae_sample_size = vae.config.sample_size #  size of pixel space
    img_tensor = utils.image_fill_mask_boundary_average(image_path=image_path, mask_path=mask_path, tensor_size=vae_sample_size, blur_radius=0, device=device, dtype=dtype) # Image to tensor
    with torch.no_grad():  # Transform image_tensor to latent space representation
        latent_original_image = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor

    # Sample noise to add to the original image latent
    noise = torch.randn_like(latents)

    # Applying the diffusion iterations.
    for i, t in enumerate(scheduler.timesteps):
        # 'prev_latent' will be the previous latent of the loop. It will not be touched thourghtout the loop.
        # 'latents' will be the current latent representation, which will be changed and pass to the next iteration.
        prev_latent = latents # Note: scale_model_input actually do nothing with current schedule, but safer. 

        with torch.no_grad():
            noise_pred_text = unet(
                prev_latent,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond = unet(
                prev_latent, 
                t, 
                encoder_hidden_states=uncond_embeddings
            ).sample

        # Apply Classifier Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step( # Denoise with Classifier Free Guidance noise
            noise_pred, t, prev_latent
        ).prev_sample

        # Forward-noise the original latent to the current timestep t
        latents_original_image_noised = scheduler.add_noise(
            latent_original_image,
            noise,
            t
        )
        
        # Get loop variables value from the 'mask_scheduler'.
        mask_latent, mask_edge_latent = mask_scheduler.get_masks(inpaint_iteration, i)
        optimization_scale = mask_scheduler.get_optimization_scale(inpaint_iteration, i)
        gamma = mask_scheduler.get_gamma(inpaint_iteration, i)

        # Apply latent space optimization with the loss function
        latents = latents.detach().requires_grad_(True)
        loss = loss_fn(
            gamma=gamma,
            diffusion_latent=latents, 
            original_image_latent=latents_original_image_noised,
            mask=mask_latent,
            mask_edge=mask_edge_latent
        )
        grad = torch.autograd.grad(loss, latents)[0]
        with torch.no_grad():
            latents = latents - optimization_scale * grad
        
        latents = latents.detach()

    # Append 'decode(latents)' to images list.
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents ).sample
    image = utils.tensor_to_image(image_tensor=image)
    images.append(image)
# Choose image with maximal CLIP score
clip_score = utils.CLIPScore(device, dtype)
CLIP_scores = [clip_score.CLIP_score(img, prompt) for img in images]
best_index = CLIP_scores.index(max(CLIP_scores))
best_CLIP_image = images[best_index]
best_CLIP_image.save(saving_output_path)