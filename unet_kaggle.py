import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from nets import Unet
from utils import TestDataset, decode_output


def go_pre(args):
    # 训练设备
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 特征网络
    print("backbone = " + args.backbone)
    # 检查保存文件夹是否存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # 生成提交csv
    sub_df = pd.DataFrame(columns=["id", "class", "predicted"])
    class_dict = dict(zip([0, 1, 2], ['large_bowel', 'small_bowel', 'stomach']))

    # 加载模型
    model = Unet(num_classes=args.num_classes * 2, pretrained=False, backbone=args.backbone).eval()

    if args.model_path != '':
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 加载dataloader
    data_list = []
    for item_case in os.listdir(args.pic_path):
        for item_day in os.listdir(os.path.join(args.pic_path, item_case)):
            path = os.path.join(args.pic_path, item_case, item_day, 'scans')
            data_list.extend(map(lambda x: os.path.join(path, x), os.listdir(path)))

    id_dict = dict(zip(range(len(data_list)), data_list))
    dataset = TestDataset(data_list, id_dict, [args.h, args.w], args.is_pre)
    gen = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # 开始预测
    with tqdm(total=len(gen), mininterval=0.3) as pbar:
        with torch.no_grad():
            model.eval().to(device)
            for item, (png, label, nw, nh, oh, ow) in enumerate(gen):
                png = png.type(torch.FloatTensor).to(device)
                output = model(png)
                nw = nw.cpu().numpy()
                nh = nh.cpu().numpy()
                ow = ow.cpu().numpy()
                oh = oh.cpu().numpy()
                label = label.cpu().numpy()
                for item_batch in range(output.shape[0]):
                    pr = torch.dstack([F.softmax(output[item_batch].permute(1, 2, 0)[..., 2 * i:2 * (i + 1)],
                                                 dim=-1) for i in range(args.num_classes)]).cpu().numpy()
                    pr = np.concatenate([np.expand_dims(pr[..., 2 * i:2 * (i + 1)].argmax(axis=-1),
                                                        -1) for i in range(args.num_classes)], axis=-1) * 255
                    pr = pr[int((args.h - nh[item_batch]) // 2): int((args.h - nh[item_batch]) // 2 + nh[item_batch]),
                         int((args.w - nw[item_batch]) // 2): int((args.w - nw[item_batch]) // 2 + nw[item_batch]),
                         :]
                    pr = cv2.resize(pr, (ow[item_batch], oh[item_batch]), interpolation=cv2.INTER_NEAREST)
                    sub_df = decode_output(pr, sub_df, id_dict[label[item_batch]][:-4])
                    # cv2.imwrite(os.path.join(args.save_dir, id_dict[label[item_batch]]), pr)
                    # png_raw = cv2.imread(os.path.join(args.pic_path, id_dict[label[item_batch]]))
                    # png_label = cv2.imread(os.path.join("./data/label_pic", id_dict[label[item_batch]]))*255
                    # overlapping = cv2.addWeighted(png_raw, 1, pr.astype(png_raw.dtype), 0.15, 0)
                    # overlapping_label = cv2.addWeighted(png_raw, 1, png_label, 0.15, 0)
                    # cv2.imwrite(os.path.join(args.save_dir, id_dict[label[item_batch]]), overlapping)
                    # cv2.imwrite(os.path.join('./data/test/label', id_dict[label[item_batch]]), overlapping_label)
                pbar.update(1)
    sub_df['class'] = sub_df['class'].apply(lambda x: class_dict[x])
    sub_df['predicted'] = sub_df['predicted'].apply(lambda x: "".join([str(i) + " " for i in x]))
    sub_df['id'] = sub_df['id'].apply(lambda x: str(x.split("/")[5]) + "_" + str(
        x.split("/")[-1].split("_")[0] + '_' + x.split("/")[-1].split("_")[1]))
    sub_df.to_csv(os.path.join(args.save_dir, 'submission.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='提交设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--num_classes', type=int, default=3, help='种类数量')
    parser.add_argument('--save_dir', type=str, default="./data/test", help='存储文件夹位置')
    parser.add_argument('--model_path', type=str,
                        default="data/weights/V3 Epoch1/ep024-f_score0.890-val_f_score0.879.pth", help='模型参数位置')
    parser.add_argument('--pic_path', type=str, default=r"D:\work\project\Kaggle_uw\data\test\train", help="pic文件夹位置")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--is_pre', type=bool, default=False, help="num_workers")
    parser.add_argument('--batch_size', type=int, default=2, help="batch_size")
    parser.add_argument('--w', type=int, default=384, help='宽')
    parser.add_argument('--h', type=int, default=384, help='高')
    args = parser.parse_args()

    go_pre(args)
