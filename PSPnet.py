import os
import os.path
import cv2
from collections import OrderedDict
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils import model_zoo
from PIL import Image
from torch.autograd import  Variable
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channel, out_channel, scales):
        super(PyramidPoolingModule,self).__init__()
        self.featureMaps = [] # saving four scales features
        for scale in scales:
            seq = nn.Sequential(
                nn.AdaptiveAvgPool2d(scale), #average pooling for scales [1,2,3,6]
                nn.Conv2d(in_channel,out_channel,kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace = True)
            ).cuda()
            self.featureMaps.append(seq)
            
    def forward(self,x):
        x_size = x.size()
        upsamplings = [x]
        upsampling_size = x_size[2:] # we only need the size of the upsampling for F.interpolate
        for feature in self.featureMaps:
            # 2D Upsampling (resizing) different scales, bilinear for 2D
            # if mode = linear(3D) or bilinear(2D), align_corners = true, else false
            upsampling = F.interpolate(feature(x), upsampling_size, mode='bilinear', align_corners=True).cuda()
            upsamplings.append(upsampling)
        concat_ = torch.cat(upsamplings, 1).cuda() # concat feature map and scaled features together
        return concat_


class PSPNet(nn.Module):
    def __init__(self, res_layers = 50, scales = (1,2,3,6), dropout = 0.1, classes = 150):
        super().__init__()
        # resnet selection
        # PPM in channel selection
        PPM_inchannel = 2048
        if res_layers == 18:
            self.feature_Map = resnet18(pretrained = True)
            PPM_inchannel = 512
        elif res_layers == 34:
            self.feature_Map = resnet34(pretrained = True)
            PPM_inchannel = 512
        elif res_layers == 101:
            self.feature_Map = resnet101(pretrained = True)
            PPM_inchannel = 2048
        elif res_layers == 152:
            self.feature_Map = resnet152(pretrained = True)
            PPM_inchannel = 2048
        else:
            self.feature_Map = resnet50(pretrained = True)
            PPM_inchannel = 2048
        self.PPM = PyramidPoolingModule(PPM_inchannel, 512, scales)
        self.output = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),
            nn.Conv2d(1024, 512, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, classes, kernel_size = 1)
        )
    def forward(self, x):
        shape = x.size()
        x = x.cuda()
        out = self.feature_Map(x)
        out = self.PPM(out[0])
        out = self.output(out).cuda()
        out = F.interpolate(out,shape[2:], mode = 'bilinear', align_corners = True ).cuda()
        return out  