import numpy as np
from PIL import Image

import torch
from torch import nn
import cv2
from reference_code.util_simple import makedir_for_file_if_not_exist


# reference from https://github.com/hszhao/semseg
# set the mean and standard variance
def set_img_mean_std():
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std

# reference from https://github.com/hszhao/semseg
# convert gray scale image to color image by using palette
def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

# reference from https://github.com/hszhao/semseg
# resize one signle image with a give size (long size)
# return a scaled image
def resize_img(image, long_size: int):
    h, w, _ = image.shape # original shape
    new_h = long_size
    new_w = long_size
    if h > w:
        new_w = round(long_size / float(h) * w)
    else:
        new_h = round(long_size / float(w) * h)
    # https://blog.csdn.net/guyuealian/article/details/85097633
    # this resource is help to understand resize and Inter_linear
    image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image_scale


# resize one signle image with a give size (long size)
# return a scaled image

def resize_img_ver2(image, interpolation=cv2.INTER_LINEAR):

    h, w, _ = image.shape
    new_h = (h - 1) // 8 * 8 + 1
    new_w = (w - 1) // 8 * 8 + 1
    image_scale = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    return image_scale

# set the image to crop_h, crop_w size and using padding to fill the broader
def crop_img_step1(image, crop_h, crop_w, mean):

    ori_h, ori_w, _ = image.shape
    # calcuate the size differencd between crop size (crop_h, crop_w) and current size (ori_h, ori_w)
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    # padding is half of difference
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        # create a border around the image like a photo frame
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)

    print('ori_h, ori_w ', ori_h, ori_w)
    print('crop_h, crop_w ', crop_h, crop_w)
    print('image shape ', image.shape)
    # cropping back to the original size
    img_crop_index = [pad_h_half, pad_h_half + ori_h, pad_w_half, pad_w_half + ori_w]  
    return img_crop_index, image

# reference from https://github.com/hszhao/semseg
# after step 1, if the size larger than (crop_h, crop_w), then crop the image as multiple images into the network
def crop_img_step2(image, crop_h, crop_w, stride_rate=2 / 3):
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))  #
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)

    img_crop_indexes = []
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h  # expected h direction starting point
            e_h = min(s_h + crop_h, new_h)  # actual h direction ending point
            s_h = e_h - crop_h   # actual starting point
            # same as previous procedure
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            img_crop_indexes.append([s_h, e_h, s_w, e_w])

    return img_crop_indexes

# reference from https://github.com/hszhao/semseg
# the function is from net_process in origional resourse.
# But, i have break that function for easier understanding.
def image_numpy2tensor(image, mean, std=None):
    # convert dimension from (H,W,c) to (C,H,W), then conver the numpy to pytorch.tensor
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    # normalization for transferred image
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    return input


# ############################################### Evaluation Metrics ###############################################
# reference from https://github.com/hszhao/semseg
# https://blog.csdn.net/lingzhou33/article/details/87901365 Some resource here could help us to understand IoU
# calculate area_intersection, area_union, area_target
def intersectionAndUnion(output, target, K, ignore_index=255):
    
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]

    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# reference from https://github.com/hszhao/semseg
# havent use this in test. It is only for training process
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]

    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# save image into direct folder
def save_img(dst_path, image):
    makedir_for_file_if_not_exist(dst_path)
    cv2.imwrite(dst_path, image)
    print('saved image to ', dst_path)
