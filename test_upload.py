import os
import requests
import cv2
import numpy as np
from PIL import Image
import io

def test_upload():
    """Test uploading an image to the detection endpoint."""
    print("Testing image upload functionality...")
    
    # Flask server URL
    base_url = "http://127.0.0.1:5000"
    
    # Look for any JPG/JPEG files in the current directory or data directory
    test_image_path = None
    
    # Search in data directory first
    for subdir in ['data/train/images', 'data/valid/images', 'data/test/images']:
        if os.path.exists(subdir):
            for file in os.listdir(subdir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image_path = os.path.join(subdir, file)
                    break
            if test_image_path:
                break
    
    # If no image found in data dir, check current directory
    if not test_image_path:
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image_path = file
                break
    
    if not test_image_path:
        print("Error: No test image found. Please place a JPG or PNG file in the current directory.")
        return
    
    print(f"Using test image: {test_image_path}")
    
    try:
        # Open and prepare the test image
        with open(test_image_path, 'rb') as f:
            files = {'file': (os.path.basename(test_image_path), f, 'image/jpeg')}
            
            print("Uploading image to detection endpoint...")
            response = requests.post(f"{base_url}/detect", files=files)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'error' in response_data:
                    print(f"Error from server: {response_data['error']}")
                    return
                
                print("Upload successful!")
                print(f"Detections: {response_data['num_detections']}")
                
                # Now try to get the original and result images
                original_url = f"{base_url}/uploads/{response_data['original']}"
                result_url = f"{base_url}/uploads/{response_data['result']}"
                
                print(f"Fetching original image from: {original_url}")
                original_response = requests.get(original_url)
                
                print(f"Fetching result image from: {result_url}")
                result_response = requests.get(result_url)
                
                if original_response.status_code == 200 and result_response.status_code == 200:
                    print("Successfully retrieved both images!")
                    
                    # Save the images locally to verify
                    with open('test_original.jpg', 'wb') as f:
                        f.write(original_response.content)
                    
                    with open('test_result.jpg', 'wb') as f:
                        f.write(result_response.content)
                    
                    print("Saved test images as 'test_original.jpg' and 'test_result.jpg'")
                else:
                    print(f"Failed to retrieve images. Original status: {original_response.status_code}, Result status: {result_response.status_code}")
            else:
                print(f"Upload failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_upload()
    print("Test complete") 