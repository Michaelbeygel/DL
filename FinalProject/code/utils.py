# This file containes helping functions for the project!
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import binary_dilation, label
import requests
import string


def create_pipeline_components(model_id, device, dtype):
    """
    Create a pipeline from a hugging face model path.
    Specifically designed for the project model.
    The pipeline components will be:
    Tokenizer, Text Encoder, Unet, VAE, Scheduler.
    Return a dictionary of (name, element)

    :param model_id: The diffusion model in use.
    :param device: The device to run the models on.
    :param dtype: The torch dtype to use.
    """
    components_dict = {}
    components_dict["tokenizer"] = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )
    
    components_dict["text_encoder"] = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype
    ).to(device)

    components_dict["unet"] = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype
    ).to(device)
    
    components_dict["vae"] = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype
    ).to(device)

    components_dict["scheduler"] = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler"
    )

    return components_dict

def get_image_and_mask(image_num=1, is_vanilla=True):
    """
    Returns the paths for the image, mask and output.
    """
    image_path = f"../data/images/image{image_num:02d}.jpg"
    mask_path = f"../data/masks/mask{image_num:02d}.jpg"
    if is_vanilla:
        output_path = f"../data/outputs/vanilla_outputs/inpaint{image_num:02d}.jpg"    
    else:
        output_path = f"../data/outputs/main_outputs/inpaint{image_num:02d}.jpg"

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    return image, mask, output_path

def get_prompt(image_num=1):
    """
    Returns prompt number 'image_num' from "promprt.txt" file.
    """
    prompt_path = "../data/prompts.txt"
    with open(prompt_path, "r") as f:
        prompt = f.read().splitlines()[image_num - 1]
    return prompt

def image_to_tensor(image, tensor_size, device, dtype):
    """
    Transfrom an image to a tensor.
    """
    img = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((tensor_size, tensor_size)),
        transforms.ToTensor(),              # [0, 1], shape [Channels, Height, Width]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to cope with autoencoder value range
    ])
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to( # Move image_tensor to GPU, and transform to dtype = torch.float16
        device=device,
        dtype=dtype
    )
    return img_tensor

def mask_to_tensor(mask, tensor_size, blur_radius, mask_shrink, device, dtype):
    """
    Transfrom a mask to a tensor.
    Choose how to transform the mask with 'blur_radius' and 'mask_shrink'.
    """
    # Create a tensor with 0-1 such that white->0, black->1
    mask = mask.convert("L")  # "L" = single channel
    mask = mask.resize((tensor_size, tensor_size))

    # Apply Gaussian blur if blur_radius > 0
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Shrink or expand mask
    if mask_shrink != 0:
        filter_size = max(1, abs(mask_shrink)*2+1)
        if mask_shrink > 0:
            mask = mask.filter(ImageFilter.MinFilter(filter_size))  # shrink white area
        else:
            mask = mask.filter(ImageFilter.MaxFilter(filter_size))  # expand white area

    mask_tensor = transforms.ToTensor()(mask)
    mask_tensor = 1.0 - mask_tensor
    mask_tensor = mask_tensor.to( # Move mask_tensor to GPU, and transform to dtype
        device=device,
        dtype=dtype
    )
    return mask_tensor

def get_edge_mask(mask_path, tensor_size, edge_boundry_size, device, dtype):
    # Create a tensor with 0-1 such that white->0, black->1
    mask = Image.open(mask_path).convert("L")  # "L" = single channel
    mask = mask.resize((tensor_size, tensor_size))

    mask_tensor = transforms.ToTensor()(mask)            # [1,H,W], values in [0,1]
    mask_tensor = 1.0 - mask_tensor
    mask_tensor = mask_tensor.to( # Move mask_tensor to GPU, and transform to dtype
        device=device,
        dtype=dtype
    )

    # Calculate the edges mask
    kernel_size = edge_boundry_size * 2 + 1
    big_mask = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=edge_boundry_size)
    small_mask = -F.max_pool2d(-mask_tensor, kernel_size=kernel_size, stride=1, padding=edge_boundry_size)
    edge_mask = big_mask - small_mask

    return edge_mask


def tensor_to_image(image_tensor):
    """
    Transform tensor to an image.
    Specifically designated to transform VAE decoder output to an image.

    :param image_tensor: The image tensor, expected in BCHW format.
    """
    image = image_tensor[0].detach().cpu().float() # Select first image in batch, move to CPU, and convert to float32.
    image = (image / 2 + 0.5).clamp(0, 1) # De-normalize from [-1, 1] to [0, 1]
    image = image.permute(1, 2, 0)        # Permute from CHW to HWC
    image = (image * 255).round().to(torch.uint8).numpy() # Scale to [0, 255]
    image = Image.fromarray(image)
    return image

def get_text_embeddings(prompt, tokenizer, text_encoder, device):
    """
    Create tokens from prompt, then create an embedding of these tokens.

    :param prompt: The text prompt to embed.
    :param tokenizer: The tokenizer model.
    :param text_encoder: The text encoder model.
    :param device: The device to run the models on.
    :return: The generated text embeddings.
    """
    # Tokenize the prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # Transform tokens to embedding
    with torch.no_grad():
        embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    return embeddings

# Latent loss for optimized latent space images.
class LatentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def non_mask_preserve_loss(self, diffusion_latent, original_image_latent, mask):
        """
        Calculates the "non mask preserve loss", which is suppose to make the non-masked areas as similar to the given image latent.
        The loss will be calculated as:

            L_2^2[(diffusion_latent * mask) - (original_image_latent * mask)]

        Where L_2^2 is the squered L_2 norm.
        
        :param diffusion_latent: The latnet space tensor of the current diffusion step.
        :param original_image_latent: The latent space tensor of the given image.
        :param mask: The binary latent space tensor of the mask.
        :return: The value of the "non mask preserve loss".
        """
        
        masked_diffusion_latent = diffusion_latent * mask
        original_given_image_latent = original_image_latent * mask

        masked_difference_latent = masked_diffusion_latent - original_given_image_latent
        # Return the (L_2)^2 norm of the difference latent space representations.
        return torch.sum(masked_difference_latent**2)

    def mask_edge_total_variation_loss(self, latent, mask_edge):
        # Create a mask that consists only of the edges of the original mask.
        latent_edge_area = latent * mask_edge # Take the area that is near the edge of the mask only
        
        diff_height = latent_edge_area[:, :, 1:, :] - latent_edge_area[:, :, :-1, :]
        diff_width = latent_edge_area[:, :, :, 1:] - latent_edge_area[:, :, :, :-1]

        total_variation_loss = torch.sum(torch.abs(diff_height)) + torch.sum(torch.abs(diff_width))
        return total_variation_loss

    def forward(self, gamma, diffusion_latent, original_image_latent, mask, mask_edge):
        non_mask_loss = self.non_mask_preserve_loss(diffusion_latent, original_image_latent, mask)
        total_variation_loss = self.mask_edge_total_variation_loss(latent=diffusion_latent, mask_edge=mask_edge)
        loss = non_mask_loss + gamma * total_variation_loss
        return loss

def image_fill_mask_boundary_average(image_path, mask_path, tensor_size, blur_radius, device, dtype):
    """
    Load image and mask, resize them, compute the average color of pixels surrounding the mask edge, and fill with it the masked region.

    :param image_path: Local path to image.
    :param mask_path: Local path to mask.
    :param tensor_size: Desired image size.
    :return: PIL.Image
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((tensor_size, tensor_size))

    # Load mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((tensor_size, tensor_size))
    # Blur the mask
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    img_np = np.array(img).astype(np.float32)
    mask_np = np.array(mask).astype(np.float32) / 255.0   # normalize to [0,1]

    # Binary mask only for component detection
    mask_bin = mask_np > 0.5
    labeled_mask, num_features = label(mask_bin)

    result = img_np.copy()

    for i in range(1, num_features + 1):

        component_mask = labeled_mask == i

        # Find boundary ring
        dilated = binary_dilation(component_mask, iterations=3)
        boundary_ring = dilated & (~component_mask)

        if not boundary_ring.any():
            boundary_ring = ~component_mask

        avg_color = img_np[boundary_ring].mean(axis=0)

        # Soft blending
        component_mask_soft = mask_np * component_mask

        for c in range(3):
            result[..., c] = (
                component_mask_soft * avg_color[c]
                + (1 - component_mask_soft) * result[..., c]
            )

    result = np.clip(result, 0, 255).astype(np.uint8)

    img = Image.fromarray(result)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=dtype)

    return img_tensor

class CLIPScore(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")                
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def CLIP_score(self, image, text_prompt):
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        return logits_per_image.sum().item()

class MaskScheduler():
    def __init__(self, timestemps, mask_path, tensor_size, device, dtype):
        self.timestemps = timestemps
        self.device = device
        self.dtype = dtype
        self.start_phase_treshold = 12
        self.regular_mask = mask_to_tensor(mask_path=mask_path, tensor_size=tensor_size, blur_radius=0, mask_shrink=0, device=device, dtype=dtype)
        self.edge_mask = get_edge_mask(mask_path=mask_path, tensor_size=tensor_size, edge_boundry_size=1, device=device, dtype=dtype)
        # Fill the masks_list variable with multiple masks
        masks_variables = [ # The first element corresponds to 'blur_radius' and the second to 'mask_shrink'
            [0,3], [0,2], [0,2], [0,1], [0,1]
        ]
        self.masks_list = []
        for blur_radius, mask_shrink in masks_variables:
            self.masks_list.append(mask_to_tensor(mask_path=mask_path, tensor_size=tensor_size, blur_radius=blur_radius, mask_shrink=mask_shrink, device=device, dtype=dtype))
        
    def get_masks(self, inpaint_iteration, i):
        if i > self.start_phase_treshold:
            return self.regular_mask, self.edge_mask
        return self.masks_list[inpaint_iteration], self.edge_mask

    def get_optimization_scale(self, inpaint_iteration, i):
        if i < self.start_phase_treshold:
            return 0.7
        return 0.4

    def get_gamma(self, inpaint_iteration, i):
        if i < self.start_phase_treshold:
            return 0.01
        return 0.005


class InpaintingEvaluator:
    def __init__(self, device, dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        # Standard fidelity metrics
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        # FID accumulates stats globally across the dataset
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        
        # Semantic alignment metric
        self.clip_scorer = CLIPScore(device, dtype) # Uses your existing CLIPScore class

    def evaluate_step(self, gen_tensor, tar_tensor, prompt, is_main=True):
        """
        Per-sample evaluation.
        gen_tensor/tar_tensor: VAE output range [-1, 1]
        """
        # Convert to [0, 1] for SSIM and PSNR
        gen_01 = (gen_tensor / 2.0 + 0.5).clamp(0, 1)
        tar_01 = (tar_tensor / 2.0 + 0.5).clamp(0, 1)
        
        # 1. Structural/Pixel metrics
        ssim_val = self.ssim_metric(gen_01, tar_01).item()
        psnr_val = self.psnr_metric(gen_01, tar_01).item()
        
        # 2. Semantic alignment
        # Convert tensor to PIL for your existing CLIPScore.CLIP_score method
        from torchvision.transforms import ToPILImage
        gen_pil = ToPILImage()(gen_01.squeeze(0).cpu())
        clip_val = self.clip_scorer.CLIP_score(gen_pil, prompt)
        
        # 3. Update FID State (Expects uint8 [0, 255])
        gen_u8 = (gen_01 * 255).to(torch.uint8)
        tar_u8 = (tar_01 * 255).to(torch.uint8)
        
        # Only update 'real' distribution once per sample pair
        if is_main:
            self.fid_metric.update(tar_u8, real=True)
        self.fid_metric.update(gen_u8, real=False)
        
        return {"PSNR": psnr_val, "SSIM": ssim_val, "CLIP": clip_val}

    def compute_final_fid(self):
        """Returns the global dataset FID score"""
        return self.fid_metric.compute().item()
