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
