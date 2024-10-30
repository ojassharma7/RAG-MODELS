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


# Tokenize a list of captions using CLIP's tokenizer
captions = ["A group of people on a beach", "Two young men playing football"]  # Example captions
text_inputs = clip.tokenize(captions).to(device)

# Encode the preprocessed image
with torch.no_grad():
    image_features = model.encode_image(preprocessed_image)

# Encode the text (captions)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# Print the shape of the encoded image and text features
print(f"Image Features Shape: {image_features.shape}")
print(f"Text Features Shape: {text_features.shape}")


import torch.nn.functional as F

# Compute cosine similarity between image features and text features
similarity = F.cosine_similarity(image_features, text_features, dim=-1)

# Print the similarity scores
print(f"Cosine Similarity Scores: {similarity}")

# Example: Retrieve the caption with the highest similarity to the image
best_caption_idx = torch.argmax(similarity).item()
best_caption = captions[best_caption_idx]
print(f"Best Matching Caption for the Image: {best_caption}")


    
