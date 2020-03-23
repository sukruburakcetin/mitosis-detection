import glob

import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from natsort import natsorted
from torch.autograd import Variable
import matplotlib.pyplot as plt

from loadImagesNew import transform_ori, transform_ori_2, test_load, test_load_2

from netNew import model, loss_fn

from torch.backends import cudnn
import random

#Run this if you want to load the model
model.load_state_dict(torch.load('mitosis_detection_model.pth'))

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
        cudnn.benchmark = True


model.to(device)

r_index = 0
right_prediction_result = 0
wrong_prediction_result = 0

mitosis_count = 0

correct = 0



# Show a batch of images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=True)


model.eval()

for i, (inputs, labels, path) in enumerate(test_load_2):

    # show images
    #imshow(torchvision.utils.make_grid(inputs))

    # Convert torch tensor to Variable
    #inputs = Variable(inputs)
    #labels = Variable(labels)

    # CUDA = torch.cuda.is_available()
    # if CUDA:
    #     inputs = inputs.cuda()
    #     labels = labels.cuda()

    inputs = inputs.to(device)
    labels = labels.to(device)

    #print("Path: " + str(path))

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)  # Calculate the loss
    loss += loss.data
    # Record the correct predictions for training data
    _, predicted = torch.max(outputs, 1)

    print(i)
    if predicted[0].item() == 0:
        p = 0
        mitosis_count = mitosis_count + 1
    else:
        p = 1
    print("Path: " + str(path) + " Predicted: " + str(p))


#print("wrong_prediction_result: " + str(wrong_prediction_result))
#print("right_prediction_result: " + str(right_prediction_result))
print("mitosis_count: " + str(mitosis_count))