import argparse
import copy
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from utils import TestDataset, decode_output, get_model, make_predict_csv


def go_pre(args):
    # 训练设备
    print("GPU: ", end="")
    print(torch.cuda.is_available())
    print("")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 特征网络
    print("backbone = ", end="")
    print(args.backbone)
    print("")

    # 权值文件
    print("model_path = ", end="")
    print(args.model_path)
    print("")

    # 是否预处理
    print("is_pre: ", end="")
    print(args.is_pre)
    print("")

    # 检查保存文件夹是否存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 生成提交csv
    sub_df = pd.DataFrame(columns=["id", "class", "predicted"])
    class_dict = dict(zip([0, 1, 2], ['large_bowel', 'small_bowel', 'stomach']))

    # 加载模型
    model_list = []
    assert len(args.backbone) == len(args.model_path)
    for item in range(len(args.backbone)):
        model = get_model(args.backbone[item], args.model_path[item], args.num_classes, args.pretrained).eval()
        model_list.append(model)

    # 获取预测csv
    class_df = make_predict_csv(args.pic_path, args.val_csv_path)

    # 生成dataloader
    dataset = TestDataset(copy.deepcopy(class_df), [args.h, args.w], args.is_pre)
    gen = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # 开始预测
    with tqdm(total=len(gen), mininterval=0.3) as pbar:
        with torch.no_grad():
            for model in model_list:
                model.eval().to(device)
            for item, (png, label_item, nw, nh, oh, ow) in enumerate(gen):
                png = png.type(torch.FloatTensor).to(device)
                nw = nw.cpu().numpy()
                nh = nh.cpu().numpy()
                ow = ow.cpu().numpy()
                oh = oh.cpu().numpy()
                label_item = label_item.cpu().numpy()
                output_class = model_list[0](png)
                for item_batch in range(label_item.shape[0]):
                    pr_class = output_class[item_batch].argmax().cpu().numpy()
                    class_df.loc[label_item[item_batch], "class_predict"] = pr_class
                    if pr_class == 0.0:
                        output = torch.dstack([torch.ones([png.shape[0], png.shape[0], png.shape[0], 1]),
                                               torch.zeros([png.shape[0], png.shape[0], png.shape[0], 1])])
                        output = torch.dstack([output, output, output])
                        # output = model_list[0](png)
                    elif pr_class == 1.0:
                        output = model_list[1](torch.unsqueeze(png[item_batch, ...], 0))
                    elif pr_class == 2.0:
                        output = model_list[2](torch.unsqueeze(png[item_batch, ...], 0))
                    pr = torch.dstack([F.softmax(output[0, ...].permute(1, 2, 0)[..., 2 * i:2 * (i + 1)],
                                                 dim=-1) for i in range(args.num_classes)]).cpu().numpy()
                    pr = np.concatenate([np.expand_dims(pr[..., 2 * i:2 * (i + 1)].argmax(axis=-1),
                                                        -1) for i in range(args.num_classes)], axis=-1) * 255
                    pr = pr[int((args.h - nh[item_batch]) // 2): int((args.h - nh[item_batch]) // 2 + nh[item_batch]),
                            int((args.w - nw[item_batch]) // 2): int((args.w - nw[item_batch]) // 2 + nw[item_batch]),
                            :]
                    pr = cv2.resize(pr, (ow[item_batch], oh[item_batch]), interpolation=cv2.INTER_NEAREST)
                    sub_df = decode_output(pr, sub_df, class_df.loc[label_item[item_batch], "id"])
                pbar.update(1)

    # 替换类别
    sub_df['class'] = sub_df['class'].apply(lambda x: class_dict[x])

    # 生成submission.csv
    if os.path.exists(os.path.join(args.pic_path, 'test')):
        df_ssub = pd.read_csv(os.path.join(args.pic_path, 'sample_submission.csv'))
        del df_ssub['predicted']
        sub_df = df_ssub.merge(sub_df, on=['id', 'class'])
        assert len(sub_df) == len(df_ssub)
    else:
        class_df.to_csv(os.path.join(args.save_dir, 'class_predict.csv'), index=False)
    sub_df[['id', 'class', 'predicted']].to_csv(os.path.join(args.save_dir, 'submission.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='提交设置')
    parser.add_argument('--backbone', type=str, nargs='+', required=True, help='特征网络选择，默认resnet50')
    parser.add_argument('--num_classes', type=int, default=3, help='种类数量')
    parser.add_argument('--save_dir', type=str, default="./", help='存储文件夹位置')
    parser.add_argument('--model_path', type=str, nargs='+', required=True, help='模型参数位置')
    parser.add_argument('--pic_path', type=str, default=r"../input/uw-madison-gi-tract-image-segmentation",
                        help="pic文件夹位置")
    parser.add_argument('--val_csv_path', type=str,
                        default=r"D:\work\project\Kaggle_uw\data\weights\class_weights\csv\val_csv.csv", help='预测csv路径')
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--is_pre', default=False, action='store_true', help="是否预处理")
    parser.add_argument('--pretrained', default=False, action='store_true', help="是否预训练")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--w', type=int, default=384, help='宽')
    parser.add_argument('--h', type=int, default=384, help='高')
    args = parser.parse_args()

    go_pre(args)
