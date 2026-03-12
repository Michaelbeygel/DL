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

class MainPipeline():
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
        self.loss_fn = utils.LatentLoss()
        self.clip_score = utils.CLIPScore(self.device, self.dtype)

    def run_main(self, image, mask, prompt):
        # Embedd the prompt. workflow: prompt -> tokens -> embedding
        text_embeddings = utils.get_text_embeddings(prompt, self.tokenizer,self.text_encoder, self.device)
        # Embedd an empty prompt for the Classifier Free Guidance
        uncond_embeddings = utils.get_text_embeddings("", self.tokenizer, self.text_encoder, self.device)

        images = []
        num_of_inpaintings = 5
        timestemps = 25
        self.scheduler.set_timesteps(timestemps)
        guidance_scale = 7 # Control the CFG. Higher values -> higher dependency on the prompt. Values should be around 6-12.

        # Set up a mask scheduler. This will organize masks and variables change thourghout the diffusion process.
        mask_scheduler = utils.MaskScheduler(
            timestemps=timestemps, 
            mask=mask, 
            tensor_size=self.latent_sample_size,
            device=self.device,
            dtype=self.dtype
        )

        # Transform image to latent space with filled mask area.
        img_tensor = utils.fill_mask_area(image=image, mask=mask, tensor_size=self.vae_sample_size, blur_radius=0, device=self.device, dtype=self.dtype) # Image to tensor
        with torch.no_grad():  # Transform img_tensor to latent space representation
            latent_original_image = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        
        for inpaint_iteration in range(num_of_inpaintings):
            # Sample random latent space data
            batch_size = 1
            latents = torch.randn(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    self.latent_sample_size,
                    self.latent_sample_size,
                ),
                device=self.device,
                dtype=self.dtype,
            ) * self.scheduler.init_noise_sigma # Scale the noisy latent by the scheduler scalar

            noise = torch.randn_like(latents)

            # Applying the diffusion iterations.
            for i, t in enumerate(self.scheduler.timesteps):
                prev_latent = latents # Note: scale_model_input actually do nothing with current schedule, but safer. 

                with torch.no_grad():
                    noise_pred_text = self.unet(
                        prev_latent,
                        t,
                        encoder_hidden_states=text_embeddings
                    ).sample
                    noise_pred_uncond = self.unet(
                        prev_latent, 
                        t, 
                        encoder_hidden_states=uncond_embeddings
                    ).sample

                # Apply Classifier Free Guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(
                    noise_pred, t, prev_latent
                ).prev_sample

                # Noise original image
                latents_original_image_noised = self.scheduler.add_noise(
                    latent_original_image,
                    noise,
                    t
                )

                # Get loop variables values from the 'mask_scheduler'.
                mask_latent, mask_edge_latent = mask_scheduler.get_masks(inpaint_iteration, i)
                optimization_scale = mask_scheduler.get_optimization_scale(inpaint_iteration, i)
                gamma = mask_scheduler.get_gamma(inpaint_iteration, i)
                # Apply latent space optimization with the loss function
                latents = latents.detach().requires_grad_(True)
                loss = self.loss_fn(
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
            latents = latents / self.vae.config.scaling_factor
            with torch.no_grad():
                image = self.vae.decode(latents).sample
            image = utils.tensor_to_image(image_tensor=image)
            images.append(image)
        # Choose image with maximal CLIP score
        CLIP_scores = [self.clip_score.CLIP_score(img, prompt) for img in images]
        best_index = CLIP_scores.index(max(CLIP_scores))
        best_CLIP_image = images[best_index]
        return best_CLIP_image
    
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inpainting with specified image number")
    parser.add_argument("--image_num", type=int, default=1, help="Image number to process (default: 1)")
    args = parser.parse_args()
    image_num = args.image_num

    main_pipeline = MainPipeline()
    image, mask, output_path = utils.get_image_and_mask(image_num=image_num, is_vanilla=False)
    prompt = utils.get_prompt(image_num=image_num)
    output_image = main_pipeline.run_main(image, mask, prompt)
    output_image.save(output_path)