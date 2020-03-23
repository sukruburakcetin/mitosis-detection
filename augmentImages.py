from imgaug import augmenters as iaa
import numpy as np
import math
from scipy import misc, ndimage
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import glob
from skimage import io

# images = io.imread_collection('C:\\Users\\Burak\\Desktop\\NewNucleiDetection\\images\\*.PNG')
images = io.imread_collection('C:\\Users\\Burak\\Desktop\\NewNucleiDetection\\masks\\*.PNG')

# for y in range(0, len(images2)):
#     images_resized = ia.imresize_single_image(images2[y], (256, 256))
#     if y != 0:
#         misc.imsave('C:\Users\Burak\Desktop\Resized\Rs' + str(y) + '.PNG', images_resized)

# images = io.imread_collection('C:\Users\Burak\Desktop\Resized\*.PNG')

flip = iaa.Fliplr(1)# always horizontally flip each input image
vflipper = iaa.Flipud(1)# vertically flip each input image with 90% probability
scaler = iaa.Affine(scale={"y": (0.8, 1.2)})# scale each input image to 80-120% on the y axis
rotate = iaa.Affine(rotate=(90, 90))
translater = iaa.Affine(translate_px={"x": -60})# move each input image by 16px to the left
blurer = iaa.GaussianBlur(3.0)# blur image 2 by a sigma of 3.0
avgblurer = iaa.AverageBlur(k=(2, 7))# blur image using local means with kernel sizes between 2 and 7
mdnblurer = iaa.MedianBlur(k=(3, 11))# blur image using local medians with kernel sizes between 2 and 7
cropper = iaa.Crop(percent=(0, 0.1))# crop images by 0-10% of their height/width
# translater1 = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}) # translate by -20 to +20 percent (per axis)
# translater2 = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})# scale images to 80-120% of their size, individually per axis
# translater3 = iaa.Affine(rotate=(-45, 45))# rotate by -45 to +45 degrees
# translater4 = iaa.Affine(shear=(-16, 16))# shear by -16 to +16 degrees
# translater5 = iaa.Affine( order=[0, 1])# use nearest neighbour or bilinear interpolation (fast)
# translater6 = iaa.Affine(cval = (0, 255))  # if mode is constant, use a cval between 0 and 255
# translater7 = iaa.Affine(mode=ia.ALL)# use any of scikit-image's warping modes (see 2nd image from the top for examples)
sharper = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))# sharpen images
embosser = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)) # emboss images
inverter = iaa.Invert(0.05, per_channel=True)# invert color channels
math_adder = iaa.Add((-10, 10), per_channel=0.5)# change brightness of images (by -10 to 10 of original value)
math_multiply = iaa.Multiply((0.5, 1.5), per_channel=0.5)# change brightness of images (50-150% of original value)
contrast_normalization = iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5)# improve or worsen the contrast
grayScale = iaa.Grayscale(alpha=(0.0, 1.0)) # grayscale the image
elasticTransformer = iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)# move pixels locally around (with random strengths)
PieceAffiner = iaa.PiecewiseAffine(scale=(0.01, 0.05))# sometimes move parts of the image around




for x in range(0, len(images)):
    images_aug1 = flip.augment_image(images[x])
    images_aug2 = vflipper.augment_image(images[x])
    images_aug3 = rotate.augment_image(images[x])
    images_aug4 = blurer.augment_image(images[x])
    images_aug5 = avgblurer.augment_image(images[x])
    images_aug6 = mdnblurer.augment_image(images[x])
    images_aug7 = cropper.augment_image(images[x])
    # images_aug8 = translater1.augment_image(images[x])
    # images_aug9 = translater2.augment_image(images[x])
    # images_aug10 = translater3.augment_image(images[x])
    # images_aug11 = translater4.augment_image(images[x])
    # images_aug12 = translater5.augment_image(images[x])
    # images_aug13 = translater6.augment_image(images[x])
    # images_aug14 = translater7.augment_image(images[x])
    images_aug15 = sharper.augment_image(images[x])
    images_aug16 = embosser.augment_image(images[x])
    images_aug17 = inverter.augment_image(images[x])
    images_aug18 = math_adder.augment_image(images[x])
    images_aug19 = math_multiply.augment_image(images[x])
    # images_aug20 = contrast_normalization.augment_image(images[x])
    # images_aug21 = grayScale.augment_image(images[x])
    images_aug22 = PieceAffiner.augment_image(images[x])
    images_aug23 = rotate.augment_image(images[x])

    # images_aug22 = elasticTransformer.augment_image(images[x])

    if x != -1:

        misc.imsave('C:\\Users\\Burak\\Desktop\\augmentedNewNucleiDetection\\3\\3_masks\\' + str(x) + '.png', images_aug3)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\VFlippedTile' + str(x) + '.PNG', images_aug2)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\ScaledTile' + str(x) + '.PNG', images_aug3)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\BluredTile' + str(x) + '.PNG', images_aug4)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\AvgBluredTile' + str(x) + '.PNG', images_aug5)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\MdnBluredTile' + str(x) + '.PNG', images_aug6)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\CroppedTile' + str(x) + '.PNG', images_aug7)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\Translated1Tile' + str(x) + '.PNG', images_aug8)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\Translated2Tile' + str(x) + '.PNG', images_aug9)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\Translated3Tile' + str(x) + '.PNG', images_aug10)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\Translated4Tile' + str(x) + '.PNG', images_aug11)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\Translated5Tile' + str(x) + '.PNG', images_aug12)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\Translated6Tile' + str(x) + '.PNG', images_aug13)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\Translated7Tile' + str(x) + '.PNG', images_aug14)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\SharpenedTile' + str(x) + '.PNG', images_aug15)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\EmbosseredTile' + str(x) + '.PNG', images_aug16)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\InvertedTile' + str(x) + '.PNG', images_aug17)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\MathAddedTile' + str(x) + '.PNG', images_aug18)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\MathMultipliedTile' + str(x) + '.PNG', images_aug19)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\ContrastNormalizedTile' + str(x) + '.PNG', images_aug20)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\grayScaledTile' + str(x) + '.PNG', images_aug21)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\PieceAffinedTile' + str(x) + '.PNG', images_aug22)
        # misc.imsave('C:\Users\Burak\Desktop\iaugmentedFiles\ElasticTransformedTile' + str(x) + '.PNG', images_aug22)
