from adain.image import load_image, prepare_image
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import numpy as np

imageName = "./images/content_dir/content_3.jpg"
newIm = load_image(imageName, size=100, crop=True)
newIm2 = load_image(imageName, size=100, crop=False)
plt.imshow(newIm)
plt.show()

plt.imshow(newIm2)
plt.show()

# Seems like for the most part, the preprocessing step works well.