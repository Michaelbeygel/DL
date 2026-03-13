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
from torchvision.transforms import ToPILImage

# Create a pipeline from a hugging face model path. Specifically designed for the project model.
# The pipeline components will be: Tokenizer, Text Encoder, Unet, VAE, Scheduler.
# Return a dictionary of (name, element)
def create_pipeline_components(model_id, device, dtype):
    components_dict = {}
    components_dict["tokenizer"] = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    components_dict["text_encoder"] = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    components_dict["unet"] = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
    components_dict["vae"] = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
    components_dict["scheduler"] = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return components_dict

# Returns image, mask and outputs path, based on the image_num and the type of pipeline(Vanilla\Main).
def get_image_and_mask(image_num, is_vanilla):
    image_path = f"../data/images/image{image_num:02d}.jpg"
    mask_path = f"../data/masks/mask{image_num:02d}.jpg"
    if is_vanilla:
        output_path = f"../data/outputs/vanilla_outputs/inpaint{image_num:02d}.jpg"    
    else:
        output_path = f"../data/outputs/main_outputs/inpaint{image_num:02d}.jpg"

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    return image, mask, output_path

# Returns prompt number image_num from promprt.txt file.
def get_prompt(image_num):
    prompt_path = "../data/prompts.txt"
    with open(prompt_path, "r") as f:
        prompt = f.read().splitlines()[image_num - 1]
    return prompt

# Transfrom a PIL image to a tensor.
def image_to_tensor(image, tensor_size, device, dtype):
    img = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((tensor_size, tensor_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device, dtype)
    return img_tensor

# Transfrom a mask to a tensor.
# Choose how to transform the mask with the 'blur_radius' and 'mask_shrink' variables.
def mask_to_tensor(mask, tensor_size, blur_radius, mask_shrink, device, dtype):
    mask = mask.convert("L") # Convert to a single channel
    mask = mask.resize((tensor_size, tensor_size))

    # Apply Gaussian blur if blur_radius > 0
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Shrink or expand mask
    if mask_shrink != 0:
        filter_size = max(1, abs(mask_shrink)*2+1)
        if mask_shrink > 0:
            mask = mask.filter(ImageFilter.MinFilter(filter_size))
        else:
            mask = mask.filter(ImageFilter.MaxFilter(filter_size))

    mask_tensor = transforms.ToTensor()(mask)
    mask_tensor = 1.0 - mask_tensor
    mask_tensor = mask_tensor.to(device, dtype)
    return mask_tensor

# Returns the mask's edge, as described in Figure 1 of our work report(dry part).
def get_edge_mask(mask, tensor_size, edge_boundry_size, device, dtype):
    mask = mask.convert("L")
    mask = mask.resize((tensor_size, tensor_size))

    mask_tensor = transforms.ToTensor()(mask)
    mask_tensor = 1.0 - mask_tensor
    mask_tensor = mask_tensor.to(device, dtype)

    # Calculates the difference between an extended mask and a shrinked mask.
    kernel_size = edge_boundry_size * 2 + 1
    big_mask = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=edge_boundry_size)
    small_mask = -F.max_pool2d(-mask_tensor, kernel_size=kernel_size, stride=1, padding=edge_boundry_size)
    edge_mask = big_mask - small_mask

    return edge_mask


# Transform torch tensor to a PIL image.
# Designated to fit for our VAE decoder tensor output.
def tensor_to_image(image_tensor):
    image = image_tensor[0].detach().cpu().float()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0)
    image = (image * 255).round().to(torch.uint8).numpy()
    image = Image.fromarray(image)
    return image

# Create tokens from prompt, then create an embedding of these tokens.
def get_text_embeddings(prompt, tokenizer, text_encoder, device):
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
# The loss function is described in part 2.2 of our work report.
class LatentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # Calculates the "non mask preserve loss".
    def non_mask_preserve_loss(self, diffusion_latent, original_image_latent, mask):
        masked_diffusion_latent = diffusion_latent * mask
        original_given_image_latent = original_image_latent * mask

        masked_difference_latent = masked_diffusion_latent - original_given_image_latent
        return torch.sum(masked_difference_latent**2)

    # Calculated the "Mask edge total variation loss".
    def mask_edge_total_variation_loss(self, latent, mask_edge):
        latent_edge_area = latent * mask_edge # Take the area that is near the edge of the mask only
        # Calculates difference between rows and columns
        diff_height = latent_edge_area[:, :, 1:, :] - latent_edge_area[:, :, :-1, :]
        diff_width = latent_edge_area[:, :, :, 1:] - latent_edge_area[:, :, :, :-1]
        # Summing the differences
        total_variation_loss = torch.sum(torch.abs(diff_height)) + torch.sum(torch.abs(diff_width))
        return total_variation_loss

    # Sum the two loss functions, with the coefficient gamma.
    def forward(self, gamma, diffusion_latent, original_image_latent, mask, mask_edge):
        non_mask_loss = self.non_mask_preserve_loss(diffusion_latent, original_image_latent, mask)
        total_variation_loss = self.mask_edge_total_variation_loss(latent=diffusion_latent, mask_edge=mask_edge)
        loss = non_mask_loss + gamma * total_variation_loss
        return loss

# Compute the average color of pixels surrounding the mask edge, and fill with it the masked region.
def fill_mask_area(image, mask, tensor_size, blur_radius, device, dtype):
    img = image.convert("RGB")
    img = img.resize((tensor_size, tensor_size))

    mask = mask.convert("L")
    mask = mask.resize((tensor_size, tensor_size))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    img_np = np.array(img).astype(np.float32)
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Binary mask only for component detection
    mask_bin = mask_np > 0.5
    labeled_mask, num_features = label(mask_bin)

    result = img_np.copy()

    for i in range(1, num_features + 1):
        component_mask = labeled_mask == i
        dilated = binary_dilation(component_mask, iterations=3)
        boundary_ring = dilated & (~component_mask)

        if not boundary_ring.any():
            boundary_ring = ~component_mask
        
        avg_color = img_np[boundary_ring].mean(axis=0)
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

# The MaskScheduler class will help organize masks and variables change throughout the diffusion pipeline.
# It's explain in part 2.2 (in page 3).
class MaskScheduler():
    def __init__(self, timestemps, mask, tensor_size, device, dtype):
        self.timestemps = timestemps
        self.device = device
        self.dtype = dtype
        self.start_phase_treshold = 12
        self.regular_mask = mask_to_tensor(mask=mask, tensor_size=tensor_size, blur_radius=0, mask_shrink=0, device=device, dtype=dtype)
        self.edge_mask = get_edge_mask(mask=mask, tensor_size=tensor_size, edge_boundry_size=1, device=device, dtype=dtype)
        # Fill the masks_list variable with multiple masks
        masks_variables = [ # The first element corresponds to 'blur_radius' and the second to 'mask_shrink'
            [0,3], [0,2], [0,2], [0,1], [0,1]
        ]
        self.masks_list = []
        for blur_radius, mask_shrink in masks_variables:
            self.masks_list.append(mask_to_tensor(mask=mask, tensor_size=tensor_size, blur_radius=blur_radius, mask_shrink=mask_shrink, device=device, dtype=dtype))
        
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
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        self.clip_scorer = CLIPScore(device, dtype)

    # Per-sample evaluation.
    # gen_tensor/tar_tensor: VAE output range [-1, 1]
    def evaluate_step(self, gen_tensor, tar_tensor, prompt, is_main=True):
        gen_01 = (gen_tensor / 2.0 + 0.5).clamp(0, 1)
        tar_01 = (tar_tensor / 2.0 + 0.5).clamp(0, 1)
        ssim_val = self.ssim_metric(gen_01, tar_01).item()
        psnr_val = self.psnr_metric(gen_01, tar_01).item()
        gen_pil = ToPILImage()(gen_01.squeeze(0).cpu())
        clip_val = self.clip_scorer.CLIP_score(gen_pil, prompt)
        gen_u8 = (gen_01 * 255).to(torch.uint8)
        tar_u8 = (tar_01 * 255).to(torch.uint8)
        if is_main:
            self.fid_metric.update(tar_u8, real=True)
        self.fid_metric.update(gen_u8, real=False)
        return {"PSNR": psnr_val, "SSIM": ssim_val, "CLIP": clip_val}

    def compute_final_fid(self):
        """Returns the global dataset FID score"""
        return self.fid_metric.compute().item()
