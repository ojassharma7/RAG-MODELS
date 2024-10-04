import torch
import clip
from PIL import Image

# Check if a GPU is available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and its preprocessing pipeline
model, preprocess = clip.load("ViT-B/32", device=device)

# Example: Preprocess an image using CLIP's preprocessing pipeline
image_path = '/path/to/flickr30k/images/1000092795.jpg'  # Replace with your image path
image = Image.open(image_path)

# Preprocess the image for CLIP
preprocessed_image = preprocess(image).unsqueeze(0).to(device)

# Print the shape of the preprocessed image
print(f"Preprocessed Image Shape: {preprocessed_image.shape}")
