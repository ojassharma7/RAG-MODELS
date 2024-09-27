import pandas as pd

# Path to your CSV file
csv_file_path = 'flickr30k/captions.csv'  # Adjust this if needed

# Load the CSV file using pandas
captions_df = pd.read_csv(csv_file_path, delimiter='|')

# Display the first few rows of the dataframe to inspect it
print(captions_df.head())