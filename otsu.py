import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)

#------------------------- Otsu Thresholding in PyTorch -------------------------

def otsu(im):
    hist = torch.histc(im, bins=256, min=0, max=255)
    hist = hist / hist.sum()
    hist = hist.to(torch.float32)

    # Compute the cumulative sum of the histogram
    P = torch.cumsum(hist, dim=0)

    # Compute the cumulative mean of intensity values
    P1 = torch.cumsum(hist * torch.arange(256, device=hist.device), dim=0)

    # Compute the global intensity mean
    mG = P1[-1]

    # Compute the between-class variance
    sigmaB2 = (mG * P - P1) ** 2
    sigmaB2[P * (1 - P) == 0] = 0  # Avoid division by zero
    sigmaB2 = sigmaB2 / (P * (1 - P) + 1e-10)  # Add a small value to avoid NaN

    # Find the intensity value that maximizes sigmaB2
    threshold = torch.argmax(sigmaB2)

    return threshold

threshold = otsu(im.to(torch.float32))

print(f'Otsu threshold: {threshold.item()}')
 
# The code above is a simple implementation of the Otsu thresholding algorithm in PyTorch. The code is self-contained and can be run in a Python environment with PyTorch installed. 
# The code reads an image from the  skimage.data  module, resizes it to half its size, and converts it to a PyTorch tensor. The  otsu  function computes the Otsu threshold of the input image. The threshold is computed by finding the intensity value that maximizes the between-class variance. The threshold is then printed to the console. 
# The code demonstrates how to implement the Otsu thresholding algorithm in PyTorch using basic PyTorch operations. 
# 3.2. Otsu Thresholding in OpenCV 
# OpenCV is a popular computer vision library that provides a wide range of image processing functions. OpenCV has built-in support for the Otsu thresholding algorithm, which makes it easy to apply Otsu thresholding to images. 
# Here is an example of how to apply Otsu thresholding to an image using OpenCV: