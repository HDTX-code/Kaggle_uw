import cv2
import numpy as np
from torch.utils.data.dataset import Dataset


class unetDataset(Dataset):
    def __int__(self, csv, num_classes, input_shape):
        super(Dataset, self).__init__()
        self.csv = csv
        self.num_classes = num_classes
        self.input_shape = input_shape

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, item):
        pic_train = cv2.imread(self.csv.loc[item, "path"])
        pic_label = np.expand_dims(cv2.imread(self.csv.loc[item, "path"], cv2.IMREAD_GRAYSCALE), axis=-1)
        pic_label[pic_label >= self.num_classes] = self.num_classes
        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[pic_label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return pic_train, pic_label, seg_labels
