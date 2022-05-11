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
        for iteration, (pic_train, pic_label, seg_labels) in enumerate(gen):
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)
                pic_train = pic_train.type(torch.FloatTensor).to(device)
                pic_label = pic_label.long().to(device)
                seg_labels = seg_labels.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            outputs = model(pic_train)
            if focal_loss:
                loss = Focal_Loss(outputs, pic_label, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pic_label, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, seg_labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, seg_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            pbar_train.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                      'f_score': total_f_score / (iteration + 1),
                                      'lr': get_lr(optimizer)})
            pbar_train.update(1)

    print('Finish Train')

    if gen_val is not None:
        print('Start Validation')
        with tqdm(total=len(gen_val), desc=f'Epoch {epoch_now + 1}/{epoch_all}', postfix=dict, mininterval=0.3) as pbar_val:
            val_loss = 0
            val_f_score = 0
            model.eval().to(device)
            with torch.no_grad():
                for iteration, (pic_train, pic_label, seg_labels) in enumerate(gen_val):
                    weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)
                    pic_train = pic_train.type(torch.FloatTensor).to(device)
                    pic_label = pic_label.long().to(device)
                    seg_labels = seg_labels.type(torch.FloatTensor).to(device)

                    outputs = model(pic_train)
                    if focal_loss:
                        loss = Focal_Loss(outputs, pic_label, weights, num_classes=num_classes)
                    else:
                        loss = CE_Loss(outputs, pic_label, weights, num_classes=num_classes)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, seg_labels)
                        loss = loss + main_dice
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, seg_labels)

                    val_loss += loss.item()
                    val_f_score += _f_score.item()

                    pbar_val.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                            'f_score': val_f_score / (iteration + 1),
                                            'lr': get_lr(optimizer)})
                    pbar_val.update(1)
        # 保存模型
        with torch.no_grad():
            print('Finish Validation')
            loss_history.append_loss(epoch_now + 1, total_loss / len(gen), val_loss / len(gen_val))
            print('Epoch:' + str(epoch_now + 1) + '/' + str(epoch_all))
            print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / len(gen), val_loss / len(gen_val)))
            if ((epoch_now + 1) % 3 == 0 or epoch_now + 1 == epoch_all) and epoch_now >= epoch_Freeze:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-f_loss%.3f-val_f_loss%.3f.pth' % (
                    (epoch_now + 1), total_f_score / len(gen), val_f_score / len(gen_val))))
    else:
        with torch.no_grad():
            print('Finish Validation')
            loss_history.append_loss(epoch_now + 1, total_loss / len(gen), 0)
            print('Epoch:' + str(epoch_now + 1) + '/' + str(epoch_all))
            print('Total Loss: %.3f' % (total_loss / len(gen)))
            if ((epoch_now + 1) % 3 == 0 or epoch_now + 1 == epoch_all) and epoch_now >= epoch_Freeze:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-f_loss%.3f.pth' % (
                    (epoch_now + 1), total_f_score / len(gen))))
