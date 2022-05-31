import copy
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
from efficientunet import get_efficientunet_b0, get_efficientunet_b7, get_efficientunet_b6, get_efficientunet_b5, \
    get_efficientunet_b4, get_efficientunet_b3, get_efficientunet_b2, get_efficientunet_b1
from nets import ResNet18, BasicBlock, Unet


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def cvtColor_cv2(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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
    ih, iw = input_size
    h, w = image.shape[:2]
    image_mask = np.ones([ih, iw, 3], dtype=image.dtype) * 128
    label_mask = np.zeros([ih, iw], dtype=label.dtype)
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


# ---------------------------------------------------#
#   加载模型
# ---------------------------------------------------#
def get_model(backbone, model_path, num_classes, pretrained):
    if backbone == 'resnet18':
        model = ResNet18(BasicBlock, num_classes=num_classes)
    elif backbone == 'efficientunet_b0':
        model = get_efficientunet_b0(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    elif backbone == 'efficientunet_b1':
        model = get_efficientunet_b1(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    elif backbone == 'efficientunet_b2':
        model = get_efficientunet_b2(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    elif backbone == 'efficientunet_b3':
        model = get_efficientunet_b3(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    elif backbone == 'efficientunet_b4':
        model = get_efficientunet_b4(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    elif backbone == 'efficientunet_b5':
        model = get_efficientunet_b5(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    elif backbone == 'efficientunet_b6':
        model = get_efficientunet_b6(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    elif backbone == 'efficientunet_b7':
        model = get_efficientunet_b7(out_channels=num_classes * 2, concat_input=True, pretrained=pretrained)
    else:
        model = Unet(num_classes=num_classes * 2, pretrained=False, backbone=backbone)
    if model_path != "":
        model = load_model(model, model_path)
    return model


def load_model(model, model_path):
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    a = {}
    no_load = 0
    for k, v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v):
                a[k] = v
            else:
                no_load += 1
        except:
            pass
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print("No_load: {}".format(no_load))
    print('Finished!')
    return model


# ---------------------------------------------------#
#   解码输出
# ---------------------------------------------------#
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_output(pr, data_csv, label):
    for item_type in range(pr.shape[-1]):
        if not (pr[..., item_type] == 0).all():
            list_item = pr[..., item_type]
            list_item[list_item != 0] = 1
            list_item = rle_encode(list_item)
            data_csv.loc[len(data_csv)] = [label, item_type, list_item]
        else:
            data_csv.loc[len(data_csv)] = [label, item_type, ""]
    return data_csv


def make_predict_csv(pic_path, val_csv_path):
    data_list = []
    class_df = pd.DataFrame(columns=["id", "path", "class_predict"])
    if os.path.exists(os.path.join(pic_path, 'test')):
        path_root = os.path.join(pic_path, 'test')
        for item_case in os.listdir(path_root):
            for item_day in os.listdir(os.path.join(path_root, item_case)):
                path = os.path.join(path_root, item_case, item_day, 'scans')
                data_list.extend(map(lambda x: os.path.join(path, x), os.listdir(path)))
        class_df["path"] = data_list
        class_df["id"] = class_df["path"].apply(lambda x: str(x.split("/")[5]) + "_" + str(
            x.split("/")[-1].split("_")[0] + '_' + x.split("/")[-1].split("_")[1]))
    else:
        val_csv = pd.read_csv(val_csv_path)
        class_df[["id", "path", "class"]] = val_csv[["id", "path", "classes"]]
    return class_df
