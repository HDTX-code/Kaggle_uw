import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from unet import Unet
import matplotlib.pyplot as plt


def go_predict(args):
    print(torch.cuda.is_available())
    unet = Unet(args.model_path, args.num_classes, args.backbone, [args.w, args.h], torch.cuda.is_available())
    image = unet.detect_image(Image.open(args.pic_path), mix_type=args.mix_type)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    print((image == 0).al())
    cv2.imwrite(os.path.join(args.save_dir, 'image.jpg'), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--backbone', type=str, default='vgg', help='特征网络选择，默认resnet50')
    parser.add_argument('--model_path', type=str, required=True, help='模型参数位置')
    parser.add_argument('--num_classes', type=int, default=4, help='种类数量 + 1')
    parser.add_argument('--w', type=int, default=512, help='宽')
    parser.add_argument('--h', type=int, default=512, help='高')
    parser.add_argument('--mix_type', type=int, default=2, help='原图与生成的图进行混合模式')
    parser.add_argument('--pic_path', type=str, required=True, help='图片路径地址')
    parser.add_argument('--save_dir', type=str, default="./", help='存储文件夹位置')
    args = parser.parse_args()

    go_predict(args)
