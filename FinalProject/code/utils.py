# This file containes helping functions for the project!

from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import torch


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

def image_to_tensor(image_path, tensor_size, device, dtype):
    """
    Transform image to a normalized tensor of size (1, 3, tensor_size, tensor_size).
    
    :param image_path: Local path to image.
    :param tensor_size: Desired size of output tensor.
    :param device: The device to move the tensor to ('cuda' or 'cpu').
    :param dtype: The torch dtype to cast the tensor to.
    """
    transform = transforms.Compose([
        transforms.Resize((tensor_size, tensor_size)),
        transforms.ToTensor(),              # [0, 1], shape [Channels, Height, Width]
        transforms.Normalize([0.5], [0.5])  # Normalize to cope with autoencoder value range
    ])
    img = Image.open(image_path).convert("RGB")
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