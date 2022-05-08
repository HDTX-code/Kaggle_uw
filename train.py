import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from nets import Unet, get_lr_scheduler, set_optimizer_lr
from utils import unetDataset, fit_one_epoch, LossHistory


def go_train(args):
    # 训练设备
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + args.backbone)

    # 检查保存文件夹是否存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载模型

    model = Unet(num_classes=args.num_classes, pretrained=False, backbone=args.backbone).train()

    if args.model_path != '':
        print('Load weights {}.'.format(args.model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 生成loss_history
    loss_history = LossHistory(args.save_dir, model, input_shape=[args.h, args.w])

    # 生成dataset
    train_csv = pd.read_csv(args.train_csv_path)
    train_dataset = unetDataset(train_csv, args.num_classes, [args.h, args.w])
    if args.val_csv_path is not None:
        val_csv = pd.read_csv(args.val_csv_path)
        val_dataset = unetDataset(val_csv, args.num_classes, [args.h, args.w])
    else:
        val_dataset = None

    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 16
    lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(args.batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(args.batch_size / nbs * args.Init_lr * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(args.momentum, 0.999),
                           weight_decay=args.weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True,
                         weight_decay=args.weight_decay)
    }[args.optimizer_type]

    # ---------------------------------------#
    #   开始冻结训练
    # ---------------------------------------#
    if args.Freeze_epoch != 0:
        model.freeze_backbone()
        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func_Freeze = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.Freeze_Epoch)
        print("-----------------Start Freeze Train-----------------")

        gen = DataLoader(train_dataset, shuffle=True, batch_size=args.Freeze_batch_size,
                         num_workers=args.num_workers)
        if args.val_csv is not None:
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.Freeze_batch_size,
                                 num_workers=args.num_workers)
        else:
            gen_val = None

        for epoch_now in range(args.Freeze_epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func_Freeze, epoch_now)
            fit_one_epoch(model=model,
                          optimizer=optimizer,
                          epoch_now=epoch_now,
                          epoch_Freeze=args.Freeze_epoch,
                          epoch_all=args.Freeze_epoch + args.UnFreeze_epoch,
                          gen=gen,
                          gen_val=gen_val,
                          save_dir=args.save_dir,
                          cls_weights=args.cls_weights,
                          device=device,
                          loss_history=loss_history,
                          num_classes=args.numclasses)
    # ---------------------------------------#
    #   开始冻结训练
    # ---------------------------------------#
    print("-----------------start UnFreeze Train-----------------")
    model.unfreeze_backbone()
    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func_UnFreeze = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.UnFreeze_Epoch)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=args.UnFreeze_batch_size,
                     num_workers=args.num_workers)
    if args.val_csv is not None:
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.UnFreeze_batch_size,
                             num_workers=args.num_workers)
    else:
        gen_val = None
    for epoch_now in range(args.Freeze_epoch, args.UnFreeze_epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func_UnFreeze, epoch_now)
        fit_one_epoch(model=model,
                      optimizer=optimizer,
                      epoch_now=epoch_now,
                      epoch_Freeze=args.Freeze_epoch,
                      epoch_all=args.Freeze_epoch + args.UnFreeze_epoch,
                      gen=gen,
                      gen_val=gen_val,
                      save_dir=args.save_dir,
                      cls_weights=args.cls_weights,
                      device=device,
                      loss_history=loss_history,
                      num_classes=args.num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--num_classes', type=int, required=True, help='种类数量')
    parser.add_argument('--save_dir', type=str, default="./logs", help='存储文件夹位置')
    parser.add_argument('--model_path', type=str, default="", help='模型参数位置')
    parser.add_argument('--w', type=int, default=512, help='宽')
    parser.add_argument('--h', type=int, default=512, help='高')
    parser.add_argument('--train_csv_path', type=str, required=True, help="训练csv")
    parser.add_argument('--val_csv_path', type=str, default=None, help="验证csv")
    parser.add_argument('--optimizer_type', type=str, default='adam', help="优化器")
    parser.add_argument('--Freeze_batch_size', type=int, default=18, help="冻结训练batch_size")
    parser.add_argument('--UnFreeze_batch_size', type=int, default=8, help="解冻训练batch_size")
    parser.add_argument('--lr_decay_type', type=str, default='cos', help="使用到的学习率下降方式，可选的有'step','cos'")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--Init_lr', type=float, default=1e-4, help="最大学习率")
    parser.add_argument('--momentum', type=float, default=0.9, help="优化器动量")
    parser.add_argument('--weight_decay', type=float, default=0, help="权值衰减，使用adam时建议为0")
    args = parser.parse_args()

    go_train(args)
