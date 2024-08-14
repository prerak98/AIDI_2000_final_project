import zipfile
import os

# Path to the zip file and extraction directory
zip_file_path = '../data/speech-emotion-recognition-ravdess-data.zip'
extract_dir = '../data/ravdess/'

# Ensure the extraction directory exists
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Unzip the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Dataset extracted to:", extract_dir)
