import os
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from natsort import natsorted
from torchvision import transforms
from torch.utils.data import Dataset


class FukudaDataset(Dataset):
    def __init__(self, datadir, data_aug, preproc=None):
        super(FukudaDataset, self).__init__()
        self.datadir = datadir
        self.augmentation = data_aug
        self.preprocessing = preproc
        self.img_list = natsorted(os.listdir(os.path.join(self.datadir, 'images')))
        self.relu = torch.nn.ReLU()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        image_path = os.path.join(self.datadir, 'images', self.img_list[idx])
        label_path = os.path.join(self.datadir, 'labels_masks', self.img_list[idx])

        try:
            image = torch.Tensor(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255.0) #! Should this go to DEVICE?
        except:
            pass
        label = torch.Tensor(cv2.imread(label_path))

        if self.augmentation: image, label = self.augment_image(image, label)

        image = torch.unsqueeze(image, dim = 0)

        label_background = torch.ones_like(label[:,:,0])*255
        label_background = label_background - label[:,:,1] - label[:,:,2]

        label[:,:,0] = label_background

        image = torch.stack([image, image, image], dim=0).squeeze(1)
        if self.preprocessing:
            image = self.preprocessing(image.permute(1,2,0))
            image = image.permute(2,0,1).type(torch.FloatTensor)
        label = label.permute((2, 0, 1))/255.0
        label = (label>0).float()

        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image = norm(image)

        return image, label

    def augment_image(self, image, label):

        probHFlip = np.random.random()
        if probHFlip > 0.5:
            image = torch.flip(image, [-1])
            label = torch.flip(label, [-3])

        probVFlip = np.random.random()
        if probVFlip > 0.5:
            image = torch.flip(image, [-2])
            label = torch.flip(label, [-2])

        probGN = np.random.random()
        if probGN > 0.5:
            image = self.gaussianNoise(image)

        # probGB = np.random.random()
        # if probGB > 0.5:
        #     image = self.gaussianBlur(image)

        probIntensity = np.random.random()
        if probIntensity > 0.5:
            scale = np.random.uniform(low=0.8, high=1.2, size=(1,))[0]
            image = image * scale
            image = image.clamp(min=0.0, max=1.0)

        image = image.clamp(min=0.0, max=1.0)
        label = label.clamp(min=0.0, max=1.0)

        return image, label

    def gaussianNoise(self, clip):

        sigma, = np.random.uniform(low=0.001, high=0.005, size=(1,)).tolist()
        clip = clip + (sigma**0.5)*torch.randn(clip.shape)
        clip = clip.clamp(min=0.0, max=1.0)

        return clip

    def gaussianBlur(self, clip):

        sigma, = np.random.uniform(low=0.5, high=1.5, size=(1,)).tolist()
        clip = clip + (sigma**0.5)*torch.randn(clip.shape)
        clip = clip.clamp(min=0.0, max=1.0)

        return clip


class hprobeDataset(Dataset):
    def __init__(self, datadir):
        super(hprobeDataset, self).__init__()
        # self.args = args
        self.datadir = datadir
        self.img_list = natsorted(os.listdir(self.datadir))

    def __len__(self):
        return len(self.img_list) - 2

    def __getitem__(self, idx):

        img_path = os.path.join(self.datadir, self.img_list[idx])
        label_path = os.path.join(self.datadir, 'labels', self.img_list[idx])

        try:
            image = torch.Tensor(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        except:
            pass
        label = torch.Tensor(cv2.imread(label_path))

        image, label = self.augmentRGBClip(image, label)

        image = torch.unsqueeze(image, dim = 0)
        label = label.permute((2, 0, 1))
        return image, label

    def augmentRGBClip(self, clip, label):
        probHFlip = np.random.random()
        if probHFlip > 0.5:
            clip = torch.flip(clip, [-1])
            label = torch.flip(label, [-1])
        # probVFlip = np.random.random()
        # if probVFlip > 0.5:
        #     clip = torch.flip(clip, [-2])
        # probGB = np.random.random()
        # if probGB > 0.5:
        #     clip = self.gaussianBlur(clip)

        probGN = np.random.random()

        if probGN > 0.5:
            clip = self.gaussianNoise(clip)

        probIntensity = np.random.random()

        if probIntensity > 0.5:
            # scale = np.random.choice([0.8, 0.9, 1.1], size = 1)[0]
            scale = np.random.uniform(low=0.8, high=1.2, size=(1,))[0]
            clip = clip * scale
            clip = clip.clamp(min=0.0, max=1.0)

        clip = clip.clamp(min=0.0, max=1.0)
        label = label.clamp(min=0.0, max=1.0)

        return clip, label


    def gaussianNoise(self, clip):

        sigma, = np.random.uniform(low=0.001, high=0.005, size=(1,)).tolist()
        clip = clip + (sigma**0.5)*torch.randn(clip.shape)
        clip = clip.clamp(min=0.0, max=1.0)

        return clip