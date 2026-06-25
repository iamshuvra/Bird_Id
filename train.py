import os
import copy
import time

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.opts import opts
from model.FBODInferenceNet import FBODInferenceBody
from utils.FBODLoss import LossFunc
from utils.FB_detector import FB_Postprocess
from utils.utils import FBObj
from utils.mAP import mean_average_precision
from datasets.dataloader.dataset_bbox import CustomDataset, dataset_collate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

num_to_english_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    return [c.strip() for c in class_names]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class LablesToResults(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def covert(self, labels_list, iteration):
        label_obj_list = []
        for batch_id in range(self.batch_size):
            labels = labels_list[batch_id]
            if labels.size == 0:
                continue
            image_id = self.batch_size * iteration + batch_id
            for label in labels:
                box = [label[i] for i in range(4)]
                label_obj_list.append(FBObj(score=1., image_id=image_id, bbox=box))
        return label_obj_list


def fit_one_epoch(largest_AP_50, net, loss_func, epoch, epoch_size, epoch_size_val,
                  gen, genval, Epoch, cuda, save_model_dir, labels_to_results, detect_post_process, scaler):
    total_loss = 0
    val_loss = 0
    start_time = time.time()

    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets, names = batch[0], batch[1], batch[2]
            if cuda:
                images = torch.from_numpy(images).to('cuda:0')
                targets = [torch.from_numpy(t).to('cuda:0') for t in targets]
            else:
                images = torch.from_numpy(images)
                targets = [torch.from_numpy(t).float() for t in targets]

            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=cuda):
                outputs = net(images)
                loss = loss_func(outputs, targets)
            if cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                total_loss += loss
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'step/s': waste_time})
            pbar.update(1)
            start_time = time.time()

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        all_label_obj_list = []
        all_obj_result_list = []
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]
            labels_list = copy.deepcopy(targets_val)
            with torch.no_grad():
                if cuda:
                    images_val = torch.from_numpy(images_val).to('cuda:0')
                    targets_val = [torch.from_numpy(t).to('cuda:0') for t in targets_val]
                else:
                    images_val = torch.from_numpy(images_val)
                    targets_val = [torch.from_numpy(t).float() for t in targets_val]
                with autocast(device_type='cuda', enabled=cuda):
                    outputs = net(images_val)
                    loss = loss_func(outputs, targets_val)
                val_loss += loss

                if (epoch + 1) >= 30:
                    all_label_obj_list += labels_to_results.covert(labels_list, iteration)
                    all_obj_result_list += detect_post_process.Process(outputs, iteration)

            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    net.train()
    if (epoch + 1) >= 30:
        AP_50, REC_50, PRE_50 = mean_average_precision(all_obj_result_list, all_label_obj_list, iou_threshold=0.5)
    else:
        AP_50, REC_50, PRE_50 = 0, 0, 0

    print('Finish Validation')
    print(f'Epoch: {epoch+1}/{Epoch}')
    print(f'Total Loss: {total_loss/(epoch_size+1):.4f} | Val Loss: {val_loss/(epoch_size_val+1):.4f} | AP_50: {AP_50:.4f} | REC_50: {REC_50:.4f} | PRE_50: {PRE_50:.4f}')

    train_loss_val = total_loss / (epoch_size + 1)
    val_loss_val   = val_loss / (epoch_size_val + 1)

    # ── Save checkpoint every epoch (replaces previous, so only 1 checkpoint kept) ──
    prev_ckpt = save_model_dir + 'last_checkpoint.pth'
    if os.path.exists(prev_ckpt):
        os.remove(prev_ckpt)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'largest_AP_50': largest_AP_50,
        'train_loss': train_loss_val,
        'val_loss': val_loss_val,
    }, prev_ckpt)
    print(f'Checkpoint saved → last_checkpoint.pth (epoch {epoch+1})')

    # ── Save best AP@50 model separately (kept forever) ──
    if AP_50 > largest_AP_50:
        largest_AP_50 = AP_50
        torch.save(model.state_dict(), save_model_dir + f'best_AP50_{AP_50:.4f}_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
        print(f'New best AP@50: {AP_50:.4f} — saved best model.')

    if (epoch + 1) >= 30:
        return train_loss_val, val_loss_val, largest_AP_50, AP_50
    else:
        return train_loss_val, val_loss_val, largest_AP_50, 0.0


# ── Loss & AP50 plot state ──────────────────────────────────────────────────
x_epoch = []
record_loss = {'train_loss': [], 'test_loss': []}
fig = plt.figure()
ax0 = fig.add_subplot(111, title="Train the FB_object_detect model")
ax0.set_ylabel('loss')
ax0.set_xlabel('Epochs')

x_ap50_epoch = []
record_ap50 = {'AP_50': []}
fig_ap50 = plt.figure()
ax1 = fig_ap50.add_subplot(111, title="Train the FB_object_detect model")
ax1.set_ylabel('ap_50')
ax1.set_xlabel('Epochs')


def draw_curve_loss(epoch, train_loss, test_loss, pic_name):
    record_loss['train_loss'].append(train_loss)
    record_loss['test_loss'].append(test_loss)
    x_epoch.append(int(epoch))
    ax0.plot(x_epoch, record_loss['train_loss'], 'b', label='train')
    ax0.plot(x_epoch, record_loss['test_loss'], 'r', label='val')
    if epoch == 2:
        ax0.legend()
    fig.savefig(pic_name)


def draw_curve_ap50(epoch, ap_50, pic_name):
    record_ap50['AP_50'].append(ap_50)
    x_ap50_epoch.append(int(epoch))
    ax1.plot(x_ap50_epoch, record_ap50['AP_50'], 'g', label='AP_50')
    if epoch == 30:
        ax1.legend()
    fig_ap50.savefig(pic_name)


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    opt = opts().parse()

    if opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise ValueError("Error! assign_method must be 'auto_assign'.")

    save_model_dir = (
        "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" +
        opt.model_input_size + "/" +
        opt.input_mode + "_" + opt.aggregation_method + "_" +
        opt.backbone_name + "_" + opt.fusion_method + "_" +
        abbr_assign_method + "_" + opt.Add_name + "/"
    )
    os.makedirs(save_model_dir, exist_ok=True)

    log_pic_name_loss = save_model_dir + "loss.jpg"
    log_pic_name_ap50 = save_model_dir + "ap50.jpg"

    config_txt = save_model_dir + "config.txt"
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as f:
            f.write(f"Input mode: {opt.input_mode}\n")
            f.write(f"Data root path: {opt.data_root_path}\n")
            f.write(f"Aggregation method: {opt.aggregation_method}\n")
            f.write(f"Backbone name: {opt.backbone_name}\n")
            f.write(f"Fusion method: {opt.fusion_method}\n")
            f.write(f"Assign method: {opt.assign_method}\n")
            f.write(f"Scale factor: {opt.scale_factor}\n")
            f.write(f"Batch size: {opt.Batch_size}\n")
            f.write(f"Data augmentation: {opt.data_augmentation}\n")
            f.write(f"Learn rate: {opt.lr}\n")

    model_input_size = (
        int(opt.model_input_size.split("_")[0]),
        int(opt.model_input_size.split("_")[1])
    )  # H, W

    Cuda = torch.cuda.is_available()
    if Cuda:
        print(f'GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('WARNING: No GPU detected, training on CPU (will be very slow)')

    train_annotation_path = "./datasets/dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"
    val_annotation_path   = "./datasets/dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_val.txt"
    train_dataset_image_path = opt.data_root_path + "images/train/"
    val_dataset_image_path   = opt.data_root_path + "images/val/"

    class_names = get_classes('model_data/classes.txt')
    num_classes = len(class_names) + 1  # +1 for background

    model = FBODInferenceBody(
        input_img_num=opt.input_img_num,
        aggregation_output_channels=opt.aggregation_output_channels,
        aggregation_method=opt.aggregation_method,
        input_mode=opt.input_mode,
        backbone_name=opt.backbone_name,
        fusion_method=opt.fusion_method
    )

    device = torch.device('cuda:0' if Cuda else 'cpu')

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    loss_func = LossFunc(
        num_classes=num_classes,
        model_input_size=(model_input_size[1], model_input_size[0]),
        scale=opt.scale_factor,
        cuda=Cuda,
        gettargets=True
    )

    detect_post_process = FB_Postprocess(batch_size=opt.Batch_size, model_input_size=model_input_size, scale=opt.scale_factor)
    labels_to_results = LablesToResults(batch_size=opt.Batch_size)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val   = len(val_lines)

    Batch_size  = opt.Batch_size
    end_Epoch   = opt.end_Epoch

    optimizer    = optim.Adam(net.parameters(), opt.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # ── Resume from last checkpoint if it exists, otherwise load pretrain ──
    last_ckpt = save_model_dir + 'last_checkpoint.pth'
    largest_AP_50 = 0
    start_Epoch   = opt.start_Epoch

    if os.path.exists(last_ckpt):
        print(f'Resuming from checkpoint: {last_ckpt}')
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_Epoch   = ckpt['epoch']
        largest_AP_50 = ckpt['largest_AP_50']
        print(f'Resumed from epoch {start_Epoch}, best AP@50 so far: {largest_AP_50:.4f}')
    elif os.path.exists(opt.pretrain_model_path):
        print('Loading pretrained weights...')
        pretrained_dict = torch.load(opt.pretrain_model_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Pretrained weights loaded.')
    else:
        print('Training from scratch.')

    train_data = CustomDataset(
        train_lines, (model_input_size[1], model_input_size[0]),
        image_path=train_dataset_image_path,
        input_mode=opt.input_mode,
        continues_num=opt.input_img_num,
        data_augmentation=opt.data_augmentation
    )
    train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True,
                                  num_workers=opt.num_workers, pin_memory=Cuda,
                                  persistent_workers=(opt.num_workers > 0),
                                  collate_fn=dataset_collate)

    val_data = CustomDataset(
        val_lines, (model_input_size[1], model_input_size[0]),
        image_path=val_dataset_image_path,
        input_mode=opt.input_mode,
        continues_num=opt.input_img_num,
        data_augmentation=False
    )
    val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True,
                                num_workers=opt.num_workers, pin_memory=Cuda,
                                persistent_workers=(opt.num_workers > 0),
                                collate_fn=dataset_collate)
    print(f'DataLoader: num_workers={opt.num_workers}, pin_memory={Cuda}, persistent_workers={opt.num_workers > 0}')

    epoch_size     = max(1, num_train // Batch_size)
    epoch_size_val = num_val // Batch_size

    scaler = GradScaler('cuda', enabled=Cuda)
    print(f'Mixed precision (AMP): {"ON" if Cuda else "OFF"}')

    for epoch in range(start_Epoch, end_Epoch):
        train_loss, val_loss, largest_AP_50, AP_50 = fit_one_epoch(
            largest_AP_50, net, loss_func, epoch,
            epoch_size, epoch_size_val,
            train_dataloader, val_dataloader,
            end_Epoch, Cuda, save_model_dir,
            labels_to_results=labels_to_results,
            detect_post_process=detect_post_process,
            scaler=scaler
        )
        if (epoch + 1) >= 2:
            draw_curve_loss(epoch + 1, train_loss.item(), val_loss.item(), log_pic_name_loss)
        if (epoch + 1) >= 30:
            draw_curve_ap50(epoch + 1, AP_50, log_pic_name_ap50)
        lr_scheduler.step()
