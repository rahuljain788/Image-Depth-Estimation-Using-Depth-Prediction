from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
# Create a DPT feature extractor
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")

# Create a DPT depth estimation model
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Specify the URL of the image to download
url = 'https://img.freepik.com/free-photo/full-length-shot-pretty-healthy-young-lady-walking-morning-park-with-dog_171337-18880.jpg?w=360&t=st=1689213531~exp=1689214131~hmac=67dea8e3a9c9f847575bb27e690c36c3fec45b056e90a04b68a00d5b4ba8990e'

# Download and open the image using PIL
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

# Use torch.no_grad() to disable gradient computation
with torch.no_grad():
    # Pass the pixel values through the model
    outputs = model(pixel_values)
    # Access the predicted depth values from the outputs
    predicted_depth = outputs.predicted_depth


# Interpolate the predicted depth values to the original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
).squeeze()

# Convert the interpolated depth values to a numpy array
output = prediction.cpu().numpy()

# Scale and format the depth values for visualization
formatted = (output * 255 / np.max(output)).astype('uint8')

# Create an image from the formatted depth values
depth = Image.fromarray(formatted)
plt.imshow(depth, interpolation='nearest')
plt.show()