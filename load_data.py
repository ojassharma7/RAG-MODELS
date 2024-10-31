import pandas as pd

# Path to your CSV file
csv_file_path = 'flickr30k/captions.csv'  # Adjust this if needed

# Load the CSV file using pandas
captions_df = pd.read_csv(csv_file_path, delimiter='|')

# Display the first few rows of the dataframe to inspect it
print(captions_df.head())

import pandas as pd
from transformers import BertTokenizer

# Path to your dataset (adjust the path to where your data is stored)
data_dir = '/path/to/flickr30k/'
csv_file_path = data_dir + 'captions.csv'

# Load the CSV file
captions_df = pd.read_csv(csv_file_path, delimiter='|')

# Group captions by image name
image_caption_map = captions_df.groupby('image_name')['comment'].apply(list).to_dict()

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize captions
def preprocess_captions(captions):
    tokenized_captions = [tokenizer(caption, return_tensors='pt', truncation=True, padding=True, max_length=64) for caption in captions]
    return tokenized_captions

# Example: Preprocess all captions for a sample image
sample_image = '1000092795.jpg'  # Replace with any image in your dataset
preprocessed_captions = preprocess_captions(image_caption_map[sample_image])

# Print the tokenized input for the first caption
print(f"Tokenized Captions for {sample_image}: {preprocessed_captions[0]['input_ids']}")


from PIL import Image
import os
import torchvision.transforms as transforms

# Path to the image directory (adjust based on your dataset location)
image_dir = data_dir + 'images/'

# Define the image preprocessing transformations (resize, normalize)
image_size = 224  # Standard size for models like CLIP
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize image
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Function to preprocess a single image
def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Apply the transformations
    image = transform(image)
    return image

# Example: Preprocess an image
sample_image_path = os.path.join(image_dir, '1000092795.jpg')  # Replace with your actual image file
preprocessed_image = preprocess_image(sample_image_path)

# Print the shape of the preprocessed image
print(f"Preprocessed Image Shape: {preprocessed_image.shape}")


print(" this is the code for the respected clip model which is going to give u a multi modal approach")