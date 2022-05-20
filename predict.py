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
    unet = Unet(args.model_path, args.num_classes, args.backbone, [args.h, args.w], torch.cuda.is_available())
    image = unet.detect_image(args.pic_path, mix_type=args.mix_type)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    print((image == 0).all())
    cv2.imshow('2', image)
    cv2.waitKey()
    # cv2.imwrite(os.path.join(args.save_dir, 'image.jpg'), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--model_path', type=str,  help='模型参数位置',
                        default='data/weights/V3 Epoch1/ep024-f_score0.890-val_f_score0.879.pth')
    parser.add_argument('--num_classes', type=int, default=3, help='种类数量')
    parser.add_argument('--w', type=int, default=384, help='宽')
    parser.add_argument('--h', type=int, default=384, help='高')
    parser.add_argument('--mix_type', type=int, default=2, help='原图与生成的图进行混合模式')
    parser.add_argument('--pic_path', type=str,  help='图片路径地址',
                        default='data/train_pic/case44_day20_slice_0095.png')
    parser.add_argument('--save_dir', type=str, default="./", help='存储文件夹位置')
    args = parser.parse_args()

    go_predict(args)
