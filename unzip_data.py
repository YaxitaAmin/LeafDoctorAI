# unzip_data.py

# unzip_data.py
import zipfile
import os

def extract_zip(zip_path, extract_to):
    """Extract the contents of a ZIP file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"ZIP file extracted to {extract_to}")

if __name__ == "__main__":
    zip_file_path = 'C:\\Users\\YAXITA\\plant-disease-detection\\plant_village.zip'  # Replace with your actual ZIP file path
    extract_to = 'C:\\Users\\YAXITA\\plant-disease-detection\\data'  # Replace with your target folder to extract the dataset

    # Ensure the target folder exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Unzip the dataset
    extract_zip(zip_file_path, extract_to)
