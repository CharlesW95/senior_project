"""
Contains a heuristic approach to determining what "patch size" a piece of clothing deserves.
This will tell us how to adjust the style weights when training the network on a training sample.
NOTE: This approach comes with the limitation that batch processing either can't happen, or each batch
needs to contain only clothing that matches a certain patch size.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread


def weigh_images(filename):
    # First, perform a center crop (image is assumed to be square)
    image = imread(filename)
    crop_size = 128
    cropped_image = center_crop(image, crop_size)
    plt.imshow(cropped_image)
    plt.show()

    # Run some heuristic over the shirt to determine patch level

    # First try - just return total variance summed across all color channels
    total_var = 0.0
    for i in range(3):
        channel = image[:, :, i]
        channel = channel.flatten()
        total_var += np.var(channel)
    return total_var

def center_crop(image, crop_size):
    image_shape = image.shape
    offset_length = crop_size/2
    x_start = image_shape[1]//2 - offset_length
    y_start = image_shape[0]//2 - offset_length
    image = image[x_start:x_start+crop_size, y_start:y_start+crop_size, :]
    return image

starting_directory = "./images/representations/"
filenames = ["style_" + str(i+1) + ".jpg" for i in range(4)]

for filename in filenames:
    print(weigh_images(starting_directory + filename))

