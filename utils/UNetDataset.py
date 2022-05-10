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
        pic_label = cv2.imread(self.csv.loc[item, "label_path"], cv2.IMREAD_GRAYSCALE)

        pic_train, pic_label = self.get_random_data(pic_train, pic_label, self.input_shape, random=False)

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
                new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
                if new_ar < 1:
                    nh = int(scale * h)
                    nw = int(nh * new_ar)
                else:
                    nw = int(scale * w)
                    nh = int(nw / new_ar)
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
        if (image == 0).all():
            image = cv2.resize(image, (iw, ih))
            label = cv2.resize(label, (iw, ih))
        else:
            image = cv2.resize(image, (iw, ih), cv2.INTER_CUBIC)
            label = cv2.resize(label, (iw, ih), cv2.INTER_NEAREST)
        return image, label
