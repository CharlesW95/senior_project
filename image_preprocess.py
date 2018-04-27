from math import floor
import numpy as np
import tensorflow as tf
from scipy.misc import imresize

def center_crop_tf(image, crop_size=256):
    image_shape = image.get_shape().as_list()
    offset_length = floor(float(crop_size/2))
    x_start = floor(image_shape[2]/2 - offset_length)
    y_start = floor(image_shape[1]/2 - offset_length)
    image = image[:, x_start:x_start+crop_size, y_start:y_start+crop_size]
    image.set_shape((3, crop_size, crop_size))
    return image

def center_crop_np(image, crop_size=256):
    image_shape = image.shape
    offset_length = floor(float(crop_size/2))
    x_start = floor(image_shape[1]/2 - offset_length)
    y_start = floor(image_shape[0]/2 - offset_length)
    image = image[y_start:y_start+crop_size, x_start:x_start+crop_size, :]

    resized_image = imresize(image, image_shape, interp='bilinear')

    return resized_image