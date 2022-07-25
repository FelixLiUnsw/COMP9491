import os
import os.path
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
import random
print(torch.cuda.is_available())
print(torch.__version__)

from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from PSPnet import PSPNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size_setting = 32
start_lr = 0.001
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

    def __getitem__(self,idx):
        img = Image.open(self.img_list[idx])
        img = img.convert('RGB')
        anno = Image.open(self.anno_list[idx])
        trans_label = transforms.Compose([
                          transforms.Resize([224,224]),
                          ])

        img = trans_label(img)
        anno = trans_label(anno)
        if random.random() > 0.5:
          trans_flip = transforms.RandomHorizontalFlip(p=1)
          img = trans_flip(img)
          anno = trans_label(anno)

        #anno = trans_label(anno)
        anno = torch.from_numpy(np.array(anno).astype('int64')-1)  # tensor([224,224])
        trans = transforms.Compose(
                            [
                            transforms.ToTensor(),
                            ]
                    )
        img = trans(img)
        return img, anno

## DataSet
train_dataset = SegmentationDataset('/content/dataset/ADEChallengeData2016',split = 'training')
val_dataset = SegmentationDataset('/content/dataset/ADEChallengeData2016',split = 'validation')
data_loader = DataLoader(train_dataset, batch_size= batch_size_setting,shuffle=False)
val_dataset = DataLoader(val_dataset, batch_size= batch_size_setting,shuffle=False)
net = PSPNet(50,(1,2,3,6),dropout = 0.1, classes = 150).to(device)
#net.load_state_dict(torch.load('epoch.pth'))
optimizer = torch.optim.Adam(net.parameters(),lr = start_lr)
scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
criterion = nn.CrossEntropyLoss(ignore_index = -1).cuda()
print("Start Traning...")
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
for epoch in range(1, 700):
    total_loss = 0
    print("---------Epoch {}-----------".format(epoch))
    for batch in tqdm(data_loader):
        images, labels = batch  # labels -->
        images, labels = images.cuda(), labels.cuda()
        preds = net(images)   #.cuda()
        loss = criterion(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    lr_ = get_lr(optimizer)
    print("total loss {} -----> learning rate {}".format(total_loss,lr_))
    torch.save(net.state_dict(), './epoch.pth')