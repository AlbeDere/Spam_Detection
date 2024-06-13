import os
import zipfile
from zipfile import ZipFile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Set the directory where you want to download the dataset
download_dir = "C:/Users/Praxis/Documents"  # Update the path to your desired directory

# Ensure the directory exists, or create it
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Download the dataset using the Kaggle API
api = KaggleApi()
api.authenticate()
api.dataset_download_files(dataset="purusinghvi/email-spam-classification-dataset", path=download_dir, unzip=True)

# After downloading, the dataset will be unzipped in the same directory
# You can then read the CSV file as follows
dataset_path = os.path.join(download_dir, 'combined_data.csv')
df = pd.read_csv(dataset_path)
df.head(10)
