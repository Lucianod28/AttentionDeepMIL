"""Pytorch dataset object that loads breast cancer dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_WIDTH = 896
IMAGE_LENGTH = 768
CROP_WIDTH = 32
CROP_LENGTH = 32

class BreastCancerBags(data_utils.Dataset):
    def __init__(self, transform, train=True):
        self.transform = transform
        self.train = train

        # read images
        images = [] # input
        malignant = [] # labels
        for filename in os.listdir('./breast_cancer_dataset'):
            if filename.endswith('.tif'):
                images.append(Image.open('breast_cancer_dataset/' + filename))
                malignant.append(int('malignant' in filename))
        # print(images)

        # crop the image into patches into a 2d array
        image_crops = []
        for image in images:
            # print(image.size)
            cropped = []
            for i in range(int(IMAGE_WIDTH / CROP_WIDTH)):
                for j in range(int(IMAGE_LENGTH / CROP_LENGTH)):
                    left = i * CROP_WIDTH
                    upper = j * CROP_LENGTH
                    right = (i + 1) * CROP_WIDTH
                    lower = (j + 1) * CROP_LENGTH
                    cropped.append(image.crop((left, upper, right, lower)))
            image_crops.append(cropped)
        # check right image sizes
        for image in image_crops:
            for crop in image:
                assert (CROP_WIDTH, CROP_LENGTH) == crop.size

        # put the images into DataLoader
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(image_crops, malignant, test_size=0.2, random_state=42)

    def __len__(self):
        if self.train:
            return len(self.X_train)
        return len(self.X_test)

    def __getitem__(self, index):
        if self.train:
            bag = self.X_train[index]
            label = self.y_train[index]
        else:
            bag = self.X_test[index]
            label = self.y_test[index]
        tensor_bag = [self.transform(image) for image in bag]
        tensor_bag = torch.stack(tensor_bag)

        return tensor_bag, label
