import torch
import clip
from PIL import Image

# Load the CLIP model and preprocessing pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Test with a sample image
image_path = "flickr30k/images/sample_image.jpg"  # Replace with your actual image path
image = Image.open(image_path)

# Preprocess the image for CLIP
preprocessed_image = preprocess(image).unsqueeze(0).to(device)

# Print the shape of the preprocessed image
print(f"Preprocessed Image Shape: {preprocessed_image.shape}")
