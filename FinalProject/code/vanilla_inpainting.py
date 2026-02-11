import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np

# Set up device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

model_id = "sd2-community/stable-diffusion-2-base"

# Setting up pipeline components:
tokenizer = CLIPTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer"
)

text_encoder = CLIPTextModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    torch_dtype=dtype
).to(device)

unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    torch_dtype=dtype
).to(device)

vae = AutoencoderKL.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=dtype
).to(device)

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)

prompt = "Portrait of a beautiful woman sitting at a Parisian cafe table, white sweater, holding a coffee cup, soft sunlight, blurred street background, high resolution, highly detailed skin, 8k, cinematic lighting."

# Tokenize the prompt
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
# Tokenize an empty prompt for the Classifier Free Guidance
uncond_text = tokenizer(
    "",
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
# Transform tokens to embedding
with torch.no_grad():
    text_embeddings = text_encoder(
        text_inputs.input_ids.to(device)
    )[0]
# Create unconditional embeddings for Classifier Free Guidance. That is, create embeddings for an empy prompt.
with torch.no_grad():
    uncond_embeddings = text_encoder(
        uncond_text.input_ids.to(device)
    )[0]

# Sample random latent space data
batch_size = 1
latent_height = latent_width = unet.config.sample_size
latents = torch.randn(
    (
        batch_size,
        unet.config.in_channels,
        latent_height,
        latent_width,
    ),
    device=device,
    dtype=dtype,
)

scheduler.set_timesteps(50)
latents = latents * scheduler.init_noise_sigma # Scale the noisy latent by the scheduler scalar, as it was trained on
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

image = (image / 2 + 0.5).clamp(0, 1)
image = (image * 255).round().to(torch.uint8)
image = image.cpu().permute(0, 2, 3, 1)  # BCHW → BHWC
image = image[0].numpy()
image = Image.fromarray(image)
image.save("generated.png")

