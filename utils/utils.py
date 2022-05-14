import copy

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def resize_cv2(image, label, input_size):
    iw, ih = input_size
    h, w = image.shape[:2]
    image_mask = np.ones([iw, ih, 3], dtype=image.dtype) * 128
    label_mask = np.zeros([iw, ih], dtype=label.dtype)
    if iw / ih < w / h:
        nw = copy.copy(iw)
        nh = int(h / w * nw)
        mask = 1
    else:
        nh = ih
        nw = int(w / h * nh)
        mask = 0
    if (image == 0).all():
        image = cv2.resize(image, (nw, nh))
        label = cv2.resize(label, (nw, nh))
    else:
        image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
        label = cv2.resize(label, (nw, nh), cv2.INTER_NEAREST)
    if mask == 1:
        image_mask[int((ih - nh) / 2):int((ih - nh) / 2) + nh, :, :] = image
        label_mask[int((ih - nh) / 2):int((ih - nh) / 2) + nh, :] = label
    else:
        image_mask[:, int((iw - nw) / 2):int((iw - nw) / 2) + nw, :] = image
        label_mask[:, int((iw - nw) / 2):int((iw - nw) / 2) + nw] = label
    return image_mask, label_mask


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


def load_model(model, model_path):
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    a = {}
    for k, v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v):
                a[k] = v
        except:
            pass
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print('Finished!')
    return model
