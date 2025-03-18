import random
import numpy as np
import torch
from skimage import data
import torch.nn.functional as F

im = data.astronaut()
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)
nbin = random.randint(32,128)

#------------------------- Color Histogram -------------------------

nbin = 125   # For debug

out = torch.tensor([])
for channel in range(3):
    quantized_pixels = torch.div(im[channel].to(torch.float32) * nbin, 256, rounding_mode='floor')
    hist = torch.histc(quantized_pixels, bins=nbin, min=0, max=nbin-1)
    out = torch.cat((out, hist), dim=0)

out = F.normalize(out, p=1, dim=0)

print(out)