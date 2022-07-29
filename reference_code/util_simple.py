import os
import torch
import cv2

# reference from https://github.com/hszhao/semseg
# ######################################## training meters ########################################
class AverageMeter(object):


    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


# ######################################## save file into folder ########################################
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def makedir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def makedir_for_file_if_not_exist(file_path):
    file_dir = os.path.abspath(os.path.dirname(file_path))
    # print('make dir for ', file_path, '  that ', file_dir)
    makedir_if_not_exist(file_dir)


# ######################################## find the port and calculation ########################################
def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


# ######################################## model saving ########################################
def load_pretrained_model(model, model_path, logger=None):

    if os.path.isfile(model_path):
        load_rec = "=> loading checkpoint '{}'".format(model_path)
        if logger:
            logger.info("=> loading checkpoint '{}'".format(model_path))
        else:
            print(load_rec)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        load_rec = "=> loaded checkpoint '{}'".format(model_path)
        if logger:
            logger.info(load_rec)
        else:
            print(load_rec)

    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))

    return model

# remove the garbage label. change the class from [1, 150] to [-1, 149], and set -1 = 255 = garbage label.
# This is my code. The original source code forget to deal with the labels in testing.
def read_label_img(label_path, b_useADK20K=True):
    gray_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
    if b_useADK20K:
        gray_label = gray_label - 1
        gray_label[gray_label == -1] = 255
    return gray_label
