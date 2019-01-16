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

class UNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UNet,self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #UNet Arch:
        #1)Encoder
        self.conv1  = nn.Conv2d(in_channels,32,7,1,3)
        self.conv2  = nn.Conv2d(32,32,7,1,3)
        
        self.conv3  = nn.Conv2d(32,64,5,1,2)
        self.conv4  = nn.Conv2d(64,64,5,1,2)
        
        self.conv5  = nn.Conv2d(64,128,3,1,1)
        self.conv6  = nn.Conv2d(128,128,3,1,1)
        
        self.conv7  = nn.Conv2d(128,256,3,1,1)
        self.conv8  = nn.Conv2d(256,256,3,1,1)
        
        self.conv9  = nn.Conv2d(256,512,3,1,1)
        self.conv10 = nn.Conv2d(512,512,3,1,1)
        
        self.conv11 = nn.Conv2d(512,512,3,1,1)
        self.conv12 = nn.Conv2d(512,512,3,1,1)
        
        #1)Decoder
        self.up_sample = nn.Upsample(scale_factor = 2, mode='bilinear')
        
        self.conv13 = nn.Conv2d(512,512,3,1,1)
        self.conv14 = nn.Conv2d(2*512,512,3,1,1)
        
        self.conv15 = nn.Conv2d(512,256,3,1,1)
        self.conv16 = nn.Conv2d(2*256,256,3,1,1)
        
        self.conv17 = nn.Conv2d(256,128,3,1,1)
        self.conv18 = nn.Conv2d(2*128,128,3,1,1)
        
        self.conv19 = nn.Conv2d(128,64,3,1,1)
        self.conv20 = nn.Conv2d(2*64,64,3,1,1)
        
        self.conv21 = nn.Conv2d(64,32,3,1,1)
        self.conv22 = nn.Conv2d(2*32,out_channels,3,1,1)
        
    def forward(self,images):
        out1 = F.leaky_relu(self.conv2(self.conv1(images)),negative_slope=0.1)
        out2 = F.avg_pool2d(out1,2)
        out3 = F.leaky_relu(self.conv4(self.conv3(out2)),negative_slope=0.1)
        out4 = F.avg_pool2d(out3,2)
        out5 = F.leaky_relu(self.conv6(self.conv5(out4)),negative_slope=0.1)
        out6 = F.avg_pool2d(out5,2)
        out7 = F.leaky_relu(self.conv8(self.conv7(out6)),negative_slope=0.1)
        out8 = F.avg_pool2d(out7,2)
        out9 = F.leaky_relu(self.conv10(self.conv9(out8)),negative_slope=0.1)
        out10 = F.avg_pool2d(out9,2)
        out11 = F.leaky_relu(self.conv12(self.conv11(out10)),negative_slope=0.1)
        out12 = F.avg_pool2d(out11,2)
        
        out13 = self.up_sample(out12)
        out14 = F.leaky_relu(self.conv14(torch.cat((self.conv13(out13),out9),1)),negative_slope=0.1)
        
        out15 = self.up_sample(out14)
        out16 = F.leaky_relu(self.conv16(torch.cat((self.conv15(out15),out7),1)),negative_slope=0.1)
        
        out17 = self.up_sample(out16)
        out18 = F.leaky_relu(self.conv18(torch.cat((self.conv17(out16),out5),1)),negative_slope=0.1)
        
        out19 = self.up_sample(out14)
        out20 = F.leaky_relu(self.conv20(torch.cat((self.conv19(out18),out3),1)),negative_slope=0.1)
        
        out21 = self.up_sample(out14)
        out22 = F.leaky_relu(self.conv22(torch.cat((self.conv21(out20),out1),1)),negative_slope=0.1)
        return out22