import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.nn.parallel
import torch.utils.data
from pre_trained_models.pspnet import PSPNet

from reference_code.util_img_deal import (
    crop_img_step1, crop_img_step2, image_numpy2tensor, set_img_mean_std
)
from reference_code.util_simple import load_pretrained_model

# this section is partially referenced from https://github.com/hszhao/semseg
# some unnessary post processing procedures has been removed.
def net_process(model, image, mean, std=None, flip=True):
    # convert image from numpy to tensor, and do the normalization
    input = image_numpy2tensor(image, mean, std=std)
    input = input.unsqueeze(0).cuda()
    # flipping. good resource to explain https://blog.csdn.net/zouxiaolv/article/details/109984318
    if flip:
        input = torch.cat([input, input.flip(3)], 0) 
    with torch.no_grad():
        output = model(input)
    # make sure the output image and input image has same size
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    # if we use flipping on input, we need to do the same procedure in output
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    # output convert numpy，(C,H,W) -> (H,W,C)
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output

# this section is also partially reference from https://github.com/hszhao/semseg
def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3):
    img_crop_index, image = crop_img_step1(image, crop_h, crop_w, mean)
    # save_img('project_images/demo_show_imgs/old_demo_single_imgs/img2.jpg', image)
    img_crop_indexes = crop_img_step2(image, crop_h, crop_w, stride_rate=stride_rate)

    # concat image and count
    new_h, new_w, _ = image.shape
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for i in range(len(img_crop_indexes)):
        [s_h, e_h, s_w, e_w] = img_crop_indexes[i]
        image_crop = image[s_h:e_h, s_w:e_w].copy()
        count_crop[s_h:e_h, s_w:e_w] += 1
        output = net_process(model, image_crop, mean, std)
        # output shape  (473, 473, 150)
        prediction_crop[s_h:e_h, s_w:e_w, :] += output
    prediction_crop /= np.expand_dims(count_crop, 2)  # get mean
    prediction_crop = prediction_crop[img_crop_index[0]:img_crop_index[1], img_crop_index[2]:img_crop_index[3]]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

def prepare_model():
    model_layers = 50
    model_classes = 150
    model_zoom_factor = 8
    model_path = 'pre_trained_models/train_epoch_100.pth'  # evaluation model path'
    model = PSPNet(layers=model_layers, classes=model_classes, zoom_factor=model_zoom_factor, pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    model = load_pretrained_model(model, model_path)
    model.eval()
    return model, model_classes

def prepare_eval_tool():
    colors_path = 'mycode/ade20k_colors.txt'
    mean, std = set_img_mean_std()
    colors = np.loadtxt(colors_path).astype('uint8')
    #print('colors ', colors.shape)
    return mean, std, colors
