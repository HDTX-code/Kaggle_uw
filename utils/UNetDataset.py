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
        pic_label = self.get_label(pic_train, self.csv.loc[item, 'segmentation_s'], self.csv.loc[item, 'segmentation_sb'],
                                   self.csv.loc[item, 'segmentation_lb'])
        pic_label[pic_label >= self.num_classes] = self.num_classes

        pic_train = cv2.resize(pic_train, self.input_shape)
        pic_train = np.transpose(cv2.cvtColor(pic_train, cv2.COLOR_BGR2RGB), [2, 0, 1])
        pic_label = cv2.resize(pic_label, self.input_shape)


        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[pic_label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return pic_train, pic_label, seg_labels

    @staticmethod
    def get_x_new(x_raw):
        if x_raw != "0":
            x_raw = x_raw.split()
            for item in range(len(x_raw)):
                x_raw[item] = int(x_raw[item])
            x_raw_1 = x_raw[::2]
            x_raw_2 = x_raw[1::2]
            x_new = []
            for item in range(len(x_raw_1)):
                for item_index in range(x_raw_2[item]):
                    x_new.append(x_raw_1[item] + item_index)
        else:
            x_new = [0]
        return x_new

    @staticmethod
    def get_new_where(x, h):
        if x != [0]:
            y = np.zeros([len(x), 2])
            for i in range(len(x)):
                y[i, 0] = x[i] // h
                y[i, 1] = x[i] % h
            return y
        else:
            return None

    @staticmethod
    def gamma_trans(img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    def get_label(self, image, segmentation_s, segmentation_sb, segmentation_lb):
        segmentation_lb = self.get_new_where(self.get_x_new(segmentation_lb), image.shape[0])
        segmentation_sb = self.get_new_where(self.get_x_new(segmentation_sb), image.shape[0])
        segmentation_s = self.get_new_where(self.get_x_new(segmentation_s), image.shape[0])
        transparent = np.zeros([image.shape[0], image.shape[0], 1], dtype=image.dtype)
        if segmentation_lb is not None:
            for j in range(segmentation_lb.shape[0]):
                transparent[int(segmentation_lb[j, 0]), int(segmentation_lb[j, 1]), 0] = 1
        if segmentation_sb is not None:
            for j in range(segmentation_sb.shape[0]):
                transparent[int(segmentation_sb[j, 0]), int(segmentation_sb[j, 1]), 0] = 2
        if segmentation_s is not None:
            for j in range(segmentation_s.shape[0]):
                transparent[int(segmentation_s[j, 0]), int(segmentation_s[j, 1]), 0] = 3
        return transparent
