# This file containes helping functions for the project!

from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import torch
import torch.nn as nn


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

def image_to_tensor(image_path, tensor_size, is_mask, device, dtype):
    """
    Transfrom an image or mask to a tensor.
    
    :param image_path: Local path to image.
    :param tensor_size: Desired size of output tensor.
    :param is_mask: True if the image is a mask.
    :param device: The device to move the tensor to ('cuda' or 'cpu').
    :param dtype: The torch dtype to cast the tensor to.
    """
    # If the image is a mask, then create a tensor with 0-1 such that white->0, black->1
    if is_mask:
        img =  mask = Image.open(image_path).convert("L")  # "L" = single channel
        img = img.resize((tensor_size, tensor_size))
        img_tensor = transforms.ToTensor()(img)            # [1,H,W], values in [0,1]
        img_tensor = (img_tensor < 0.5).float()            # Threshold to 0/1
    else:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((tensor_size, tensor_size)),
            transforms.ToTensor(),              # [0, 1], shape [Channels, Height, Width]
            transforms.Normalize([0.5], [0.5])  # Normalize to cope with autoencoder value range
        ])
        img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to( # Move image_tensor to GPU, and transform to dtype = torch.float16
        device=device,
        dtype=dtype
    )
    return img_tensor
    
def tensor_to_image(image_tensor, saving_path):
    """
    Transform tensor to an image and saves it in given path.
    Specifically designated to transform VAE decoder output to an image.

    :param image_tensor: The image tensor, expected in BCHW format.
    :param saving_path: The path to save the image.
    """
    image = image_tensor[0].detach().cpu().float() # Select first image in batch, move to CPU, and convert to float32.
    image = (image / 2 + 0.5).clamp(0, 1) # De-normalize from [-1, 1] to [0, 1]
    image = image.permute(1, 2, 0)        # Permute from CHW to HWC
    image = (image * 255).round().to(torch.uint8).numpy() # Scale to [0, 255]
    image = Image.fromarray(image)
    image.save(saving_path)

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

    def forward(self, diffusion_latent, original_image_latent, mask):
        return self.non_mask_preserve_loss(diffusion_latent, original_image_latent, mask)