import os
import gdown
import zipfile

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# URLs for the Google Drive Folders to download
folder_urls = [
    # 'https://drive.google.com/drive/folders/1RCXsKyWzc_mHY0FX_4F_q9R1Aa1jsban?usp=drive_link', # DAIR-V2X-C content
    # 'https://drive.google.com/drive/folders/1FlBbtJfuoEOc0ey9wkkU1lGr5JtS4-gc?usp=drive_link', # DAIR-V2X-I content
    'https://drive.google.com/drive/folders/1I9Du70tQ-6s0qIXfYsIHd7uBW--BwaFJ?usp=drive_link'  # DAIR-V2X-V content
]

# Loop through each folder URL
for folder_url in folder_urls:
    print(f"\nProcessing folder: {folder_url}")

    # Extract folder ID from the URL
    folder_id = None
    if 'folders/' in folder_url:
        folder_id = folder_url.split('folders/')[1].split('?')[0]

    if not folder_id:
        print(f"  Could not extract folder ID from URL: {folder_url}")
        continue # Skip to the next URL if ID extraction fails

    # Define paths based on the folder ID
    zip_output_path = os.path.join('data', f'{folder_id}.zip')
    # Extract into a subdirectory named after the folder ID
    extract_path = os.path.join('data', folder_id)

    print(f"  Attempting to download folder (ID: {folder_id}) to {zip_output_path}...")

    try:
        # Download the folder as a zip file
        gdown.download_folder(id=folder_id, output=zip_output_path, quiet=False, use_cookies=False)
        print(f"  Successfully downloaded folder to {zip_output_path}")

        # Create the extraction directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        # Extract the downloaded zip file
        print(f"  Extracting {zip_output_path} to {extract_path}...")
        with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"  Successfully extracted folder contents to {extract_path}")

        # Optional: Remove the zip file after extraction
        print(f"  Removing temporary zip file: {zip_output_path}")
        os.remove(zip_output_path)
        print(f"  Removed {zip_output_path}")

    except Exception as e:
        print(f"  Failed to download or extract folder (ID: {folder_id}). Error: {e}")
        # Optionally keep the zip file on failure for debugging
        # if os.path.exists(zip_output_path):
        #     print(f"  Keeping potentially incomplete zip file: {zip_output_path}")

print("\nAll folder download and extraction processes finished.")
