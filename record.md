# Kaggle_uw

## V1

### Epoch_1

#### Model

+ backbone: Vgg16
+ Unet

#### Data

+ 使用了全部的数据
+ 做了gamma矫正
+ 没做数据增强
+ 没有排除垃圾数据

#### Train

+ Optimizer =  Adam
+ lr_decay_type = cos, max_lr = 1e-4, min_lr = 1e-6
+ Freeze_batch_size = 10,  UnFreeze_batch_size = 8
+ Freeze_epoch = 8, UnFreeze_epoch = 16
+ 256 * 256
+ val_per = 0.2, 随机选取

#### Train_result

+ Total_Loss = 0.303,  Val_Loss = 0.318
+ f_score = 0.759,  f_score_val = 0.744

#### Problem

+ 预处理数据集出错，[w, h]变成 了[w, w]，标注的时候也弄反了h,  w
+ 黑色背景太多，虽然f_score不错，但是因为把背景也算在了类别里，大多数图像预测出来全黑

### Epoch_2

#### Change

+ 用各个类别（包括背景）的面积的倒数除以面积的倒数的总和作为交叉熵loss权重
+ 解决epoch_1里的预处理问题