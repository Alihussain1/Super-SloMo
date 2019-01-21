import torch
import torchvision
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import math
import os
import random
from PIL import Image

class dataset_loader():
    def __init__(self,dataset_path,transforms = None):
        self.data_path = dataset_path
        self.transforms = transforms
        if not os.path.exists(dataset_path):
            raise(RuntimeError("path doesn't exist !"))
        self.frames = self.get_paths()
    def get_paths(self):
        images = []
        dirs = os.listdir(self.data_path)
        for folder in dirs:
            folder_path = os.path.join(self.data_path,folder)
            if not os.path.isdir(folder_path):
                continue
            folder_images = []
            for img in os.listdir(folder_path):
                folder_images.append(os.path.join(folder_path,img))
            images.append(folder_images)
        return images
    def __getitem__(self,index):
        frame_0_indx = random.randint(0,len(self.frames[index]) - 9) #9 = 1(index base = 0 not 1) + 8(7 frames between I0,I1)
        frame_1_indx = frame_0_indx + 8
        frame_t_indx = random.randint(frame_0_indx + 1 ,frame_1_indx - 1)
        image_0 = Image.open(self.frames[index][frame_0_indx])
        image_t = Image.open(self.frames[index][frame_t_indx])
        image_1 = Image.open(self.frames[index][frame_1_indx])
        if self.transforms is not None:
            image_0 = self.transforms(image_0)
            image_t = self.transforms(image_t)
            image_1 = self.transforms(image_1)
        return (image_0,image_t,image_1),frame_t_indx - frame_0_indx - 1
    def __len__(self):
        return len(self.frames)