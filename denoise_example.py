"""
Denoising example

This example walks through every step needed to denoise a 2D image. 
Extending this to denoise a 3D image should not need to many changes

The parameters used here is not particulary good and as you see from
the plot the denoised image doesn't look very nice. Try changing the
variables in CAPS to see how they changes the denoising result as well
as the running time.
"""

# First we import what we need
from __future__ import print_function

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import seaborn as sns
from ksvd import ksvd, denoise, create_patches, reconstruct_patches

# Utilities to make the setup shorter
from utils import center, psnr, initial_dictionary

# For timeing the code execution
import timeit

# Read the noisy image
noisy_image = misc.imread('images/lena_noisy.png').astype(float)

# Save the image size for later use. m is height, n width
m, n = noisy_image.shape

# Size of image patches.
# Try different values and see how this changes
# the quality of the denoised image
SIZE = 7
patch_size = (SIZE, SIZE)

# We want all overlapping image patches so we set this as None
max_patches = None

# Convert the image into a matrix of image patches
# Each column in the matrix 'image_patches' correspond
# to one image patch. Shape (num_elements_in_patch, n_patches),
# In this case (12*12, n_patches)
image_patches = create_patches(noisy_image, patch_size, max_patches)

# Remove the mean from the patches for better results
image_patches, mean = center(image_patches)

# The dictionary and image_patches needs the same number of rows
rows = image_patches.shape[0]

# Number of dictionary atoms. This can technically by any number, buy
# 2*rows is usually a good choice
N_ATOMS = 23
dictionary = initial_dictionary(rows, N_ATOMS)
# dictionary = np.load('dictionary.npy')

# Number of K-SVD iterations
ITERS = 2

# Number of nonzero coeffs to use in K-SVD sparse coding step
N_NONZERO = 1

# Start timing 
t1 = timeit.default_timer()

# Train the dictionary using K-SVD
dictionary = ksvd(image_patches, dictionary, ITERS, N_NONZERO)

# Variance of the noice
SIGMA = 271

# Remove the noise from the image patches
denoised_patches = denoise(image_patches, dictionary, SIGMA)

# Stop timing
t2 = timeit.default_timer()

# Add back the mean values
denoised_patches += mean

# Transform patches back into an image. The
# overlapping regions are averaged
denoised = reconstruct_patches(denoised_patches, (m, n))

original = misc.imread('images/lena.png').astype(float)

# Get signal to noise ratios
noise = psnr(original, noisy_image, 255)
nonoise = psnr(original, denoised, 255)

plt.subplot(232)
plt.title('Original')
plt.imshow(original, cmap=plt.cm.bone)
plt.axis('off')

plt.subplot(234)
plt.imshow(noisy_image, cmap=plt.cm.bone)
plt.title('Noisy, PSNR = %.2f' % noise)
plt.axis('off')

plt.subplot(236)
plt.imshow(denoised, cmap=plt.cm.bone)
plt.title('Denoised, PSNR = %.2f' % nonoise)
plt.axis('off')

plt.show()

print('PSNR noisy image:', noise)
print('PSNR denoised image:', nonoise)
print('Time:', t2 - t1)
