# This file containes helping functions for the project!

from PIL import Image
from torchvision import transforms
import torch

def image_to_tensor(image_path, tensor_size):
    """
    Transform image to a normalized tensor of size (1, 3, tensor_size, tensor_size).
    
    :param image_path: Local path to image.
    :param tensor_size: Desired size of output tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((tensor_size, tensor_size)),
        transforms.ToTensor(),              # [0, 1], shape [Channels, Height, Width]
        transforms.Normalize([0.5], [0.5])  # Normalize to cope with autoencoder value range
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def tensor_to_image(image_tensor, saving_path):
    """
    Transform tensor to an image and saves it in given path.
    Specifically designated to transform VAE decoder output to an image.

    :param image_tensor: The image tensor.
    :param saving_path: The path to save the image.
    """
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = (image * 255).round().to(torch.uint8)
    image = image.cpu().permute(0, 2, 3, 1)  # BCHW → BHWC
    image = image[0].numpy()
    image = Image.fromarray(image)
    image.save(saving_path)