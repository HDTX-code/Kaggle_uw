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
+ f_score = 0.759,  f_score_val = 0.744（未去除背景类）

#### Problem

+ 预处理数据集出错，[w, h]变成 了[w, w]，标注的时候也弄反了h,  w
+ 黑色背景太多，虽然f_score不错，但是因为把背景也算在了类别里，大多数图像预测出来全黑

### Epoch_2

#### Change

+ 用各个类别（包括背景）的面积的倒数除以面积的倒数的总和作为交叉熵loss权重
+ 解决epoch_1里的预处理问题

#### Train

+ Optimizer =  Adam
+ lr_decay_type = cos, max_lr = 1e-4, min_lr = 1e-6
+ Freeze_batch_size = 10,  UnFreeze_batch_size = 8
+ Freeze_epoch = 8, UnFreeze_epoch = 22
+ 256 * 256
+ val_per = 0.2, 随机选取

#### Train_result

+ f_score = 0.351,  f_score_val = 0.347（未去除背景类）

#### Problem

+ loss绝对值太小，在e-5量级
+ 添加权重以后，模型不再识别到全黑的区域，但仍然不准确，大多识别在边缘亮区

### Epoch_3

#### Change

+ 增加了数据增强模块，包括随机缩放，旋转，锐化，模糊
+ 进入模型训练时，对图像添加除以最大值的0-1化操作，缩小绝对误差，避免识别集中在亮区

#### Train

+ Optimizer =  Adam
+ lr_decay_type = cos, max_lr = 1e-4, min_lr = 1e-6
+ Freeze_batch_size = 10,  UnFreeze_batch_size = 8
+ Freeze_epoch = 0, UnFreeze_epoch = 24
+ 256 * 256
+ 使用Epoch_2的训练验证划分

#### Problem

+ 需要更好的前期数据预处理
+ 对dice loss和f score去除掉背景类的影响，加到total loss里
+ 需要利用上更多的信息
+ 寻找医疗影像的Unet模型预训练权重（现在还是用的voc数据集预训练的权重）

## V2

### Epoch_1

#### Change

+ 用了医疗影像数据的预训练权重
+ 对dice loss加了权重，与交叉熵的权重一样（总面积的倒数）
+ 对f_score去除了背景类的影响
+ 将backbone更换为resnet50

#### Train

+ Optimizer =  Adam
+ lr_decay_type = cos, max_lr = 1e-4, min_lr = 1e-6
+ Freeze_batch_size = 10,  UnFreeze_batch_size = 8
+ Freeze_epoch = 0, UnFreeze_epoch = 24
+ 256 * 256
+ 使用V1_Epoch_2的训练验证划分