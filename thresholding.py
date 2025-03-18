import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)
val = random.randint(0, 255)

#------------------------- Thresholding -------------------------

val = torch.ones(im.shape) * val
out = im.to(torch.float32)

out = torch.where(out > val, 255, 0)
out = out.to(torch.uint8)

#------------------------- Alternative -------------------------

out = torch.zeros(im.shape)

out[im > val] = 255

out = out.to(torch.uint8)