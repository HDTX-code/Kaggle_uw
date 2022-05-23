import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model, optimizer, epoch_now, epoch_Freeze, num_classes,
                  epoch_all, gen, gen_val, save_dir, cls_weights, device,
                  loss_history, focal_loss=True, dice_loss=True):
    print('Start Train')
    with tqdm(total=len(gen), desc=f'Epoch {epoch_now + 1}/{epoch_all}', postfix=dict, mininterval=0.3) as pbar_train:
        total_loss = 0
        total_f_score = 0
        model.train().to(device)
        for iteration, (pic_train, seg_labels) in enumerate(gen):
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)
                pic_train = pic_train.type(torch.FloatTensor).to(device)
                seg_labels = seg_labels.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            outputs = model(pic_train)
            loss = 0
            main_dice = 0
            _f_score = 0
            if focal_loss:
                for i in range(num_classes):
                    loss += Focal_Loss(outputs[:, 2*i:2*(i+1), ...], seg_labels[..., i].long(),
                                       weights[[0, i+1]], num_classes=num_classes)

            for i in range(num_classes):
                loss += CE_Loss(outputs[:, 2*i:2*(i+1), ...], seg_labels[..., i].long(),
                                weights[[0, i+1]], num_classes=num_classes)

            if dice_loss:
                for i in range(num_classes):
                    main_dice += Dice_loss(outputs[:, 2*i:2*(i+1), ...], seg_labels[..., i:i+1])
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                for i in range(num_classes):
                    _f_score += f_score(outputs[:, 2*i:2*(i+1), ...], seg_labels[..., i:i+1])
                _f_score /= num_classes

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            pbar_train.set_postfix(**{'l': total_loss / (iteration + 1),
                                      's': total_f_score / (iteration + 1),
                                      'r': get_lr(optimizer)})
            pbar_train.update(1)

    print('Finish Train')

    if gen_val is not None:
        print('Start Validation')
        with tqdm(total=len(gen_val), desc=f'Epoch {epoch_now + 1}/{epoch_all}', postfix=dict, mininterval=0.3) as pbar_val:
            val_loss = 0
            val_f_score = 0
            model.eval().to(device)
            with torch.no_grad():
                for iteration, (pic_train, seg_labels) in enumerate(gen_val):
                    weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)
                    pic_train = pic_train.type(torch.FloatTensor).to(device)
                    seg_labels = seg_labels.type(torch.FloatTensor).to(device)

                    outputs = model(pic_train)
                    loss = 0
                    main_dice = 0
                    _f_score = 0
                    if focal_loss:
                        for i in range(num_classes):
                            loss += Focal_Loss(outputs[:, 2 * i:2 * (i + 1), ...], seg_labels[..., i].long(),
                                               weights[[0, i+1]], num_classes=num_classes)

                    for i in range(num_classes):
                        loss += CE_Loss(outputs[:, 2 * i:2 * (i + 1), ...], seg_labels[..., i].long(),
                                        weights[[0, i+1]], num_classes=num_classes)

                    if dice_loss:
                        for i in range(num_classes):
                            main_dice += Dice_loss(outputs[:, 2 * i:2 * (i + 1), ...], seg_labels[..., i:i + 1])
                        loss = loss + main_dice

                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    for i in range(num_classes):
                        _f_score += f_score(outputs[:, 2 * i:2 * (i + 1), ...], seg_labels[..., i:i + 1])
                    _f_score /= num_classes

                    val_loss += loss.item()
                    val_f_score += _f_score.item()

                    pbar_val.set_postfix(**{'l': val_loss / (iteration + 1),
                                            's': val_f_score / (iteration + 1),
                                            'r': get_lr(optimizer)})
                    pbar_val.update(1)
        # 保存模型
        with torch.no_grad():
            print('Finish Validation')
            loss_history.append_loss(epoch_now + 1, total_loss / len(gen), val_loss / len(gen_val))
            print('Epoch:' + str(epoch_now + 1) + '/' + str(epoch_all))
            print('Total Loss: %.6f || Val Loss: %.6f ' % (total_loss / len(gen), val_loss / len(gen_val)))
            if ((epoch_now + 1) % 3 == 0 or epoch_now + 1 == epoch_all) and epoch_now >= epoch_Freeze:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-f_score%.3f-val_f_score%.3f.pth' % (
                    (epoch_now + 1), total_f_score / len(gen), val_f_score / len(gen_val))))
    else:
        with torch.no_grad():
            print('Finish Validation')
            loss_history.append_loss(epoch_now + 1, total_loss / len(gen), 0)
            print('Epoch:' + str(epoch_now + 1) + '/' + str(epoch_all))
            print('Total Loss: %.6f' % (total_loss / len(gen)))
            if ((epoch_now + 1) % 3 == 0 or epoch_now + 1 == epoch_all) and epoch_now >= epoch_Freeze:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-f_score%.3f.pth' % (
                    (epoch_now + 1), total_f_score / len(gen))))
