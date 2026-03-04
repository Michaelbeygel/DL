import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


# CLIP Score: Quantitative measure of Semantic Alignment (Text-to-Image similarity).
# Desired Result: HIGHER is better.
# 
# Interpretation: 
# Represents how successfully the UNet transformed the text prompt into pixels. 
# A high score indicates the 'dog' and 'grass' look like the concepts described. 
# Note: Sophisticated blending can slightly lower this score as the model 
# compromises between 'perfect prompt matching' and 'perfect background blending.'
def calculate_clip_score(image_path, prompt, device="cuda"):
    # Load the official CLIP model and processor from OpenAI, forcing the secure format
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Load the generated image
    image = Image.open(image_path).convert("RGB")

    # Process the inputs (text and image) 
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    
    # Move inputs to the correct device (GPU/CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Calculate the embeddings and the similarity score
    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image is the raw similarity score
        clip_score = outputs.logits_per_image.item() 

    return clip_score

if __name__ == "__main__":
    prompt = "A dog sitting, on grass."
    image_num = 1

    # Measure your baseline (vanilla pipeline)
    baseline_score = calculate_clip_score("../data/outputs/vanilla_outputs/inpaint0{image_num}.jpg", prompt)
    print(f"Baseline CLIP Score: {baseline_score:.2f}")

    # Measure your improved pipeline (gradient guidance + blur)
    improved_score = calculate_clip_score("../data/outputs/main_outputs/inpaint0{image_num}", prompt)
    print(f"Improved Method CLIP Score: {improved_score:.2f}")
