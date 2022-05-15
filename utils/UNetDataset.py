import copy
import math

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset


class UNetDataset(Dataset):
    def __init__(self, csv, num_classes, input_shape):
        super(Dataset, self).__init__()
        self.csv = csv
        self.num_classes = num_classes
        self.input_shape = input_shape

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, item):
        pic_train = cv2.imread(self.csv.loc[item, "path"])
        # pic_train = self.gamma_trans(pic_train, math.log10(0.5) / math.log10(np.mean(pic_train[pic_train > 0]) / 255))
        pic_label = cv2.imread(self.csv.loc[item, "label_path"], cv2.IMREAD_GRAYSCALE)

        pic_train, pic_label = self.get_random_data(pic_train, pic_label, self.input_shape, random=True)

        pic_label[pic_label >= self.num_classes] = self.num_classes
        pic_train = np.transpose(cv2.cvtColor(pic_train, cv2.COLOR_BGR2RGB), [2, 0, 1])

        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[pic_label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return pic_train, pic_label, seg_labels

    @staticmethod
    def gamma_trans(img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    @staticmethod
    def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a

    @staticmethod
    def cvtColor(image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image

    @staticmethod
    def ImageNew(src):
        blur_img = cv2.GaussianBlur(src, (0, 0), 5)
        usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
        result = usm
        return result

    @staticmethod
    def Image_GaussianBlur(img):
        kernel_size = (5, 5)
        sigma = 1.5
        img = cv2.GaussianBlur(img, kernel_size, sigma)
        return img

    @staticmethod
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

    def get_random_data(self, image, label, input_shape, jitter=.3, random=True):
        image = self.cvtColor(image)
        h, w = image.shape[0], image.shape[1]
        ih, iw = input_shape
        if random:
            #   生成随机数，scale负责随机缩放、锐化、高斯模糊，scale flip 负责上下左右旋转
            scale = self.rand(0, 1)
            scale_flip = self.rand(0, 1)
            #   随机缩放、锐化、高斯模糊
            if scale < 0.25:
                new_ar = iw / ih * self.rand(1, 2) / self.rand(1, 2)
                nh = h
                nw = int(h * new_ar)
                if (image == 0).all():
                    image = cv2.resize(image, (nw, nh))
                    label = cv2.resize(label, (nw, nh))
                else:
                    image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
                    label = cv2.resize(label, (nw, nh), cv2.INTER_NEAREST)
            elif 0.25 <= scale < 0.5:
                image = self.Image_GaussianBlur(image)
            elif 0.5 <= scale < 0.75:
                image = self.ImageNew(image)

            #   随机旋转
            if scale_flip < 0.25:
                image = cv2.flip(image, -1)
                label = cv2.flip(label, -1)
            elif 0.25 <= scale_flip < 0.5:
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)
            elif 0.5 <= scale_flip < 0.75:
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)

        #   将图像多余的部分加上灰条
        image, label = self.resize_cv2(image, label, input_shape)
        return image, label
