# Kaggle_uw

## V1

### Model

+ backbone: Vgg16
+ Unet

### Data

+ 使用了全部的数据
+ 做了gamma矫正
+ 没做数据增强
+ 没有排除垃圾数据

### Train

+ Optimizer =  Adam
+ lr_decay_type = cos, max_lr = 1e-4, min_lr = 1e-6
+ Freeze_batch_size = 10,  UnFreeze_batch_size = 8
+ Freeze_epoch = 8, UnFreeze_epoch = 16
+ 256 * 256
+ val_per = 0.2, 随机选取

### Train_result

+ Total_Loss = 0.303,  Val_Loss = 0.318
+ f_score = 0.759,  f_score_val = 0.744

### Test



### Problem