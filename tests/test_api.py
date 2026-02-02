"""
Script to test the StegaDNA Universal Controller API.
"""
import requests
import os

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.json()}")

def test_stamp_text():
    print("\nTesting Text Stamping...")
    data = {
        "user_id": "anton_vice_1337",
        "raw_text": "This is a confidential technical specification for StegaDNA."
    }
    response = requests.post(f"{BASE_URL}/stamp/text", data=data)
    print(f"Response: {response.json()}")

def test_stamp_image():
    print("\nTesting Image Stamping...")
    # Using a dummy image if exists, or just checking the endpoint
    image_path = "/Users/antonvice/Documents/programming/StegaDNA/data/images/archive_part_001/10006104631428706595_handwritten_math.jpg"
    
    if os.path.exists(image_path):
        files = {"file": open(image_path, "rb")}
        data = {"user_id": "cfo_grably_01"}
        response = requests.post(f"{BASE_URL}/stamp/image", files=files, data=data)
        print(f"Response: {response.json()}")
    else:
        print("Test image not found, skipping.")

if __name__ == "__main__":
    try:
        test_health()
        test_stamp_text()
        test_stamp_image()
    except Exception as e:
        print(f"Connection failed: {e}. Is the server running?")
