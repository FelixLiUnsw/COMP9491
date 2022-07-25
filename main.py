from net import hrnetv2, C1
import torch
import torch.nn as nn
import os
from mit_semseg.utils import find_recursive
from dataset import TestDataset
from segmodule import SegmentationModule
from mit_semseg.lib.nn import async_copy_to
from mit_semseg.lib.utils import as_numpy
from mit_semseg.utils import colorEncode
import numpy as np
from scipy.io import loadmat
import csv
from PIL import Image
from metric import metrics
import cv2
from mit_semseg.utils import accuracy
from crf import DenseCRF


colors = loadmat('./color150.mat')['colors']
names = {}
with open('./object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(data, pred):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    img_test = cv2.imread('./ADEChallengeData2016/annotations/validation/ADE_val_00000001.png', cv2.IMREAD_GRAYSCALE)
    img_test = np.array(torch.from_numpy(np.array(img_test)).long() - 1)

    acc, pix = accuracy(img_test, pred)

    print(acc)


    #print(img.shape, pred_color.shape)
    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)


    img_name = info[0].split('\\')[-1]
    Image.fromarray(im_vis).save(
        os.path.join('./result/', img_name.replace('.jpg', '.png')))






encoder = hrnetv2()
checkpoint = torch.load('./encoder_epoch_30.pth')
encoder.load_state_dict(checkpoint)

decoder = C1(num_class=150, fc_dim=720, use_softmax=True)
checkpoint2 = torch.load('./decoder_epoch_30.pth')
decoder.load_state_dict(checkpoint2)

crit = nn.NLLLoss(ignore_index=-1)

seg_module = SegmentationModule(encoder, decoder, crit).cuda()

if os.path.isdir('.\ADEChallengeData2016\images\\validation'):
    imgs = find_recursive('.\ADEChallengeData2016\images\\validation')
else:
    imgs = ['.\ADEChallengeData2016\images\\validation']
assert len(imgs), "imgs should be a path to image (.jpg) or directory."
list_test = [{'fpath_img': x} for x in imgs]

dataset_test = TestDataset(list_test)
loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False)







seg_module.eval()
torch.cuda.set_device(0)

for batch_data in loader_test:
    #batch_data = batch_data[0]
    batch_data['img_ori'] = batch_data['img_ori'].reshape(batch_data['img_ori'].shape[1:])
    segSize = (batch_data['img_ori'].shape[0],
               batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    with torch.no_grad():
        scores = torch.zeros(1, 150, segSize[0], segSize[1])
        scores = async_copy_to(scores, 0)

        for img in img_resized_list:
            img = img.reshape(img.shape[1:])
            feed_dict = batch_data.copy()
            feed_dict['img_data'] = img
            del feed_dict['img_ori']
            del feed_dict['info']
            feed_dict = async_copy_to(feed_dict, 0)

            # forward pass

            pred_tmp = seg_module(feed_dict, segSize=segSize)
            scores = scores + pred_tmp / len((300, 375, 450, 525, 600))

        #print(scores, scores.shape)
        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

        crf = DenseCRF(5, 3, 1, 4, 67, 3)

        imgg = np.array(batch_data['img_ori'])
        #imgg = np.transpose(imgg, (2, 0, 1))


        scc = np.array(scores.squeeze(0).cpu())

        pred2 = crf(imgg, scc)

        label = np.argmax(pred2, axis=0)


    # visualization
    visualize_result(
        (batch_data['img_ori'], batch_data['info']),
        pred
    )
    visualize_result(
        (batch_data['img_ori'], batch_data['info']),
        label
    )
