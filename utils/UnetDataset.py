import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class UnetDataset(Dataset):
    def __int__(self, csv):
        super(UnetDataset, self).__init__()
        self.csv = csv

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, item):
        pic_train = cv2.imread(self.csv.loc[item, "path"])
        pic_label = np.expand_dims(cv2.imread(self.csv.loc[item, "path"], cv2.IMREAD_GRAYSCALE), axis=-1)
