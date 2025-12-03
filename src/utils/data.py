import os.path as osp

import PIL.Image as PImage
import json
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
    val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
    # 打印 0, 1, 325, 999 对应的文件夹名（ImageNet WNID）
    print(f"ID 0 maps to folder: {train_set.classes[0]}")
    print(f"ID 325 maps to folder: {train_set.classes[325]}")
    print(f"ID 999 maps to folder: {train_set.classes[999]}")
    file_id_to_model_id = {
        file_system_id: model_id
        for model_id, file_system_id in enumerate(train_set.classes)
    }
    
    # 2. 将查找表写入 JSON 文件
    # 文件名：id_map_file_to_model.json
    map_file_path = osp.join(data_path, 'id_map_file_to_model.json')
    
    try:
        # 使用 json.dump 写入，indent=4 使文件易读
        with open(map_file_path, 'w') as f:
            json.dump(file_id_to_model_id, f, indent=4)
            
        # 验证您之前观察到的错误映射
        if '391' in file_id_to_model_id:
             print(f"[ID Check] File System ID '391' maps to Model ID: {file_id_to_model_id['391']}")
             
        print(f"\n[ID Map] Successfully saved File System ID to Model ID map to: {map_file_path}")
        
    except Exception as e:
        # 捕获异常，例如权限问题
        print(f"\n[ID Map Error] Failed to write map file {map_file_path}: {e}")

    # =======================================================
    
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
