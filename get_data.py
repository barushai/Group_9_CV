import urllib.request
import os

url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
filename = "dashcam.mp4"

print("Downloading video data...")
try:
    urllib.request.urlretrieve(url, filename)
    print(f"Download successful! Saved as: {filename}")
except Exception as e:
    print(f"Error during download: {e}")