# Some imports we will need for this notebook
import numpy as np
from PIL import Image
import glob
from natsort import natsorted
import cv2

# Reading and sorting the image paths from the directories
# ground_truth_images = natsorted(glob.glob('C:/Users/Burak/Desktop/train_tupac_mitosis_detection/mitoses_orj/*.jpg'))
ground_truth_images = natsorted(glob.glob('E:/dusenberrymw/deep-histopath/data/mitoses/patches/val/normal/*.jpg'))
collated_images_and_masks = list(zip(ground_truth_images))

images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]
width = int(100) # target width
height = int(100) # target height
dim = (width, height)
for r_index in range(len(images)):
   images[r_index][0] = cv2.cvtColor(images[r_index][0], cv2.COLOR_RGB2BGR)
   # cv2.imwrite('C:/Users/Burak/Desktop/train_tupac_mitosis_detection/mitosis_0/' + str(r_index) + ".jpg",  cv2.resize(images[r_index][0], dim,  interpolation = cv2.INTER_AREA ))
   cv2.imwrite('E:/dusenberrymw/deep-histopath/data/mitoses/patches/val/normal_resized/' + str(r_index) + ".jpg",  cv2.resize(images[r_index][0], dim,  interpolation = cv2.INTER_AREA ))