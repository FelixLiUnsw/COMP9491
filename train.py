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
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from glob import glob
import numpy as np
print(torch.cuda.is_available())
print(torch.__version__)

from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from PSPnet import PSPNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 16
start_lr = 0.01
trans = transforms.Compose(
                    [
                    transforms.Resize([224,224]),
                    transforms.ToTensor(),
                    ]
            )
class SegmentationDataset(Dataset):
    def __init__(self, root_folder_path, split) -> None:
        super().__init__()
        """
        split: training/validation
        """
        self.img_list = sorted(glob(os.path.join(root_folder_path,"images", split, "*.jpg")))
        self.anno_list = sorted(glob(os.path.join(root_folder_path,"annotations", split, "*.png")))
    def __len__(self):
        return len(self.img_list)
        # if self.split == '/training':
        #     return len(self.img_list)
        # else:
        #     return len(self.anno_list)

    def __getitem__(self,idx):
        img = Image.open(self.img_list[idx])
        anno = Image.open(self.anno_list[idx])
        trans = transforms.Compose(
                            [
                            transforms.Resize([224,224]),
                            transforms.ToTensor(),
                            ]
                    )
        img = trans(img)
        anno = trans(anno)
        return img, anno

## DataSet
train_dataset = SegmentationDataset('ADEChallengeData2016',split = 'training')
#val_dataset = SegmentationDataset('ADEChallengeData2016',split = 'validation')
data_loader = DataLoader(train_dataset, batch_size= 16,shuffle=False)
#val_dataset = DataLoader(val_dataset, batch_size= 16,shuffle=False)
net = PSPNet(50,(1,2,3,6),dropout = 0.1, classes = 150).to(device)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(),lr = start_lr)
scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.CrossEntropyLoss()
print("Start Traning...")
for epoch in range(1, 3):
    total_loss = 0
    print("---------Epoch {}-----------",epoch)
    for batch in tqdm(data_loader):
        images, labels = batch

        preds = net(images.cuda())

        #labels = labels.astype(np.int64)
        #labels = F.one_hot(labels.to(torch.int64), 151)
        labels = torch.argmax(labels, dim=1)
        # print(preds.shape)
        # print(labels.shape)

        loss = criterion(preds,labels)
        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    print(total_loss)