import sys
import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils import load_image

prompt = sys.argv[1]

img_path = "../images/image02.jpg"
mask_path = "../images/mask02.jpg"

init_image = load_image(img_path).resize((512, 512))
mask_image = load_image(mask_path).resize((512, 512))

repo_id = "sd2-community/stable-diffusion-2-base"
## TODO: 
##   1) Implement each scheduler step.
##   2) Apply DDPM style technique to each step.
pipe = StableDiffusionInpaintPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=7.5, num_inference_steps=25).images[0]

image.save("Inpaint02.png")