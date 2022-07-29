import os
import time
from traceback import print_tb

import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from reference_code import transform, dataset
from .model_evaluation_tool import scale_process, net_process, prepare_eval_tool, prepare_model
from reference_code.util_img_deal import (
    colorize, resize_img, intersectionAndUnion, save_img, resize_img_ver2, resize_img
)
from reference_code.util_simple import AverageMeter, makedir_if_not_exist, read_label_img



def test_single_img():
    # test file
    image_path = 'ADEChallengeData2016/images/validation/ADE_val_00000001.jpg'
    label_path = 'ADEChallengeData2016/annotations/validation/ADE_val_00000001.png'
    gray_label = read_label_img(label_path)
    print('gray_label max min avg ', np.max(gray_label), np.min(gray_label), np.mean(gray_label))

    # ############################ Setting of the image ############################
    mean, std, colors = prepare_eval_tool()
    model, model_classes = prepare_model()
    scales = [1.0]
    base_size = 512
    crop_h, crop_w = 473, 473  # set the crop size of the image
    b_doCropPred = False
    # ############################ Execution Test ############################
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
    save_img('demo_img/single_img/img0.jpg', image)

    h, w, _ = image.shape
    if b_doCropPred:
        print('image.shape ', image.shape)
        prediction = np.zeros((h, w, model_classes), dtype=float)
        # scaling
        for scale in scales:
            long_size = round(scale * base_size)
            image_scale = resize_img(image, long_size)
            print('image_scale shape ', image_scale.shape)
            save_img('demo_img/single_img/img1.jpg', image_scale)
            prediction += scale_process(model, image_scale, model_classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
    else:
        # directly use model to pred the image
        image = resize_img_ver2(image)  #  assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        prediction = net_process(model, image, mean, std, flip=False)
        print('image shape ', image.shape)
        prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)

    prediction = np.argmax(prediction, axis=2)  # shape is (H,W,150)
    print('prediction max min avg ', np.max(prediction), np.min(prediction), np.mean(prediction))

    # ############################ save the image ############################
    gray = np.uint8(prediction)
    color = colorize(gray, colors)
    # image_name = image_path.split('/')[-1].split('.')[0]
    image_name = os.path.basename(image_path)[:-4]
    #gray_path = 'project_images/demo_show_imgs/new_demo_single_imgs/gray_img_{}.png'.format(image_name)
    #color_path = 'project_images/demo_show_imgs/new_demo_single_imgs/color_img_{}.png'.format(image_name)
    gray_path = 'demo_img/single_img/gray_img.png'
    color_path = 'demo_img/single_img/color_img.png'
    print('color_path ', color_path)
    save_img(gray_path, gray)
    color.save(color_path)
    print("=> Prediction saved in {}".format(color_path))


def test_multi_imgs(test_data):
    # ############################ Parmas Setting ############################
    mean, std, colors = prepare_eval_tool()
    model, classes = prepare_model()
    scales = [1.0]
    base_size = 512
    crop_h, crop_w = 473, 473  # set crop size
    b_doCropPred = False

    # save image to here
    # dst_gray_folder = 'project_images/demo_show_imgs/new_demo_multi_imgs/gray'
    # dst_color_folder = 'project_images/demo_show_imgs/new_demo_multi_imgs/color'
    dst_gray_folder ='demo_img/test_img/gray'
    dst_color_folder ='demo_img/test_img/color'
    print('dst_gray_folder  ',dst_gray_folder)
    print('dst_color_folder  ',dst_color_folder) 

    # ############################ dataloader ############################
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    data_list = test_data.data_list

    # ############################ execute ############################
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()  # record load image time
    batch_time = AverageMeter()  # record pred time
    model.eval()
    end = time.time()
    for i, (input, _) in enumerate(test_loader):
        data_time.update(time.time() - end)

        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape

        if b_doCropPred:
            prediction = np.zeros((h, w, classes), dtype=float)
            for scale in scales:
                long_size = round(scale * base_size)
                image_scale = resize_img(image, long_size)
                prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
                prediction /= len(scales)
        else:
            # directly predict the result
            image = resize_img_ver2(image)  # assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
            prediction = net_process(model, image, mean, std, flip=False)
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)

        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            print('Test: [{}/{}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).\n'
                  .format(i + 1, len(test_loader), data_time=data_time, batch_time=batch_time))

        # ############################ save the result ############################
        makedir_if_not_exist(dst_gray_folder)
        makedir_if_not_exist(dst_color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]

        # image_name = image_path.split('/')[-1].split('.')[0]
        image_name = os.path.basename(image_path)[:-4]
        print('image_path ',image_path)
        print('image_name ',image_name)

        gray_path = os.path.join(dst_gray_folder, image_name + '.png')
        color_path = os.path.join(dst_color_folder, image_name + '.png')
        print('dst_gray_folder ',dst_gray_folder)
        print('gray_path ',gray_path)
        save_img(gray_path, gray)
        color.save(color_path)

# this code is partially reference from https://github.com/hszhao/semseg/blob/master
# The source code forget to remove the garbage label.
# the class is in range [1,150], where 0 is garbage class
# but the author just use the 150 classes to do the segmentation..
# the accuracy is poor if you directly execute the code
# so I add a function to reduce 1 for each label, the label in range [-1,149]
# and then, set -1 class to 255 as what we did in pretrained model.

# The source code has lots of issues.. So I rewrite most of part. and check every single progress in his post-processing
# some post processing is unreasonable.
# you can have a look on first image in test_img
# there are some pink color in brown region. The part should all brown region.
# if you just use the model to predict the image. and use softmax and argmax.
# You will find it is a good prediction, because only very little pink region in there
# However, after his post processing, the result is bad in first image. 
# But amazingly we can get the color of tree in front of the house(my team's model cannot find it)
# I think his post processing could be improved somehow.
# I have modify his post processing by my method. The pink region is smaller than his post processing
# I think its flipping and cropping issue. 
# The author try to crop image to multiple small image to feed model. Which could improve the execution time, but reduce accuarcy
def cal_acc(data_list, pred_folder, classes, names):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for i, (image_path, target_path) in enumerate(data_list):
        image_name = os.path.basename(image_path)[:-4]
        pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
        target = read_label_img(target_path)
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        print('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name + '.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)  # shape is (150,)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)  # shape is (150,)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)



    valid_class = union_meter.sum > 0
    mIoU = np.mean(iou_class[valid_class])
    mAcc = np.mean(accuracy_class[valid_class])

    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        if valid_class[i]:
            print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

def cal_multi_imgs_acc(test_data, classes=150):

    dst_gray_folder ='demo_img/test_img/gray'
    names_path = 'mycode/ade20k_names.txt'

    names = [line.rstrip('\n') for line in open(names_path)]
    cal_acc(test_data.data_list, dst_gray_folder, classes, names)


def main(test = 'acc'):
    #
    data_root = 'ADEChallengeData2016'
    test_list = 'ADEChallengeData2016/ADE20K_validation.txt'
    test_transform = transform.Compose([transform.ToTensor()])  # numpy 转 tensor
    test_data = dataset.SemData(split='train', data_root=data_root, data_list=test_list, transform=test_transform)
    if test == 'pred':
        test_multi_imgs(test_data)
    elif test == 'acc':
        cal_multi_imgs_acc(test_data, classes=150)
    elif test == 'single':
        test_single_img()
if __name__ == '__main__':
    main()
