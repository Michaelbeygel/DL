import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
import utils

class vanillaPipeline():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.model_id = "sd2-community/stable-diffusion-2-base"
        # Set up pipeline components
        self.pipe = utils.create_pipeline_components(self.model_id, self.device, self.dtype)
        self.tokenizer, self.text_encoder = self.pipe["tokenizer"], self.pipe["text_encoder"]
        self.unet, self.vae, self.scheduler = self.pipe["unet"], self.pipe["vae"], self.pipe["scheduler"]
        self.latent_sample_size = self.unet.config.sample_size
        self.vae_sample_size = self.vae.config.sample_size

    def run_vanilla(self, image, mask, prompt):
        # Transform image to torch tensor and then to latent space
        img_tensor = utils.image_to_tensor(image=image, tensor_size=self.vae_sample_size, device=self.device, dtype=self.dtype) # Image to tensor
        with torch.no_grad():  # Transform image_tensor to latent space representation
            latent_original_image = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor

        # Transform mask to latent space size binary torch tensor based on the masked region.
        mask_latent_tensor = utils.mask_to_tensor(mask=mask, tensor_size=self.latent_sample_size, blur_radius=0, mask_shrink=0, device=self.device, dtype=self.dtype)

        # Embedd the prompt. workflow: prompt -> tokens -> embedding
        text_embeddings = utils.get_text_embeddings(prompt, self.tokenizer,self.text_encoder, self.device)
        # Embedd an empty prompt for the Classifier Free Guidance
        uncond_embeddings = utils.get_text_embeddings("", self.tokenizer,self.text_encoder, self.device)

        # Sample random latent space data
        batch_size = 1
        latents = torch.randn(
            (
                batch_size,
                self.unet.config.in_channels,
                self.latent_sample_size, # Height = latent sample size
                self.latent_sample_size, # Widrh = latent sample size
            ),
            device=self.device,
            dtype=self.dtype,
        ) * self.scheduler.init_noise_sigma # Scale the noisy latent by the scheduler scalar, as it was trained on

        self.scheduler.set_timesteps(25)
        guidance_scale = 7 # Control the CFG. Higher values -> higher dependency on the prompt. Values should be around 6-12.

        # Applying the diffusion iterations.
        for t in self.scheduler.timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t) # Note: scale_model_input actually do nothing with current schedule, but safer. 

            with torch.no_grad():
                noise_pred_text = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                noise_pred_uncond = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=uncond_embeddings
                ).sample

            # Apply Classifier Free Guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

            # ---- INPAINTING PART ----
            # Forward-noise the original latent to the current timestep
            # "Mimics" the noise at timestep t and apply it to the original image(in the tesnor latent repressentation)
            noise = torch.randn_like(latents)
            latents_original_image_noised = self.scheduler.add_noise(
                latent_original_image,
                noise,
                t
            )
            
            # Clamp known region. Note: '*' is elemnent-wise multiplication.
            latents = mask_latent_tensor * latents_original_image_noised + (1 - mask_latent_tensor) * latents

        latents = latents / self.vae.config.scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = utils.tensor_to_image(image_tensor=image)
        return image
    
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inpainting with specified image number")
    parser.add_argument("--image_num", type=int, default=1, help="Image number to process (default: 1)")
    args = parser.parse_args()
    image_num = args.image_num

    vanilla_pipeline = vanillaPipeline()
    image, mask, output_path = utils.get_image_and_mask(image_num=image_num, is_vanilla=True)
    prompt = utils.get_prompt(image_num=image_num)
    output_image = vanilla_pipeline.run_vanilla(image, mask, prompt)
    output_image.save(output_path)
