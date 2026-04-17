"""
单通道图像+3通道mask

改进等比例缩放 + pad 到 256×256
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F  # 导入torch.nn.functional
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from PIL import ImageOps


def resize_with_aspect_and_pad(img, target_size, is_mask=False, fill=0):
    target_h, target_w = target_size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = Image.NEAREST if is_mask else Image.BILINEAR
    img = img.resize((new_w, new_h), interp)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    img = ImageOps.expand(img, border=(left, top, right, bottom), fill=fill)
    return img

class MyDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'training',plane = False, num_classes=3):
        df = pd.read_csv(os.path.join(data_path, 'mydata_' + mode + '_groundtruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        # modify
        self.num_classes = num_classes

        # 定义颜色到类别的映射
        self.color_map = {
            (0, 0, 0): 0,        # 背景 - 黑色
            (128, 0, 0): 1,      # 类别1 - 红色
            (0, 128, 0): 2,      # 类别2 - 绿色
        }

    def __len__(self):
        return len(self.name_list)
    
    def rgb_to_mask(self, rgb_mask):
        """将RGB mask转换为类别mask"""
        rgb_mask = np.array(rgb_mask)
        h, w = rgb_mask.shape[:2]
        mask = np.zeros((h, w), dtype=np.int64)
        
        for color, class_id in self.color_map.items():
            # 计算颜色距离，允许小幅偏差
            diff = np.abs(rgb_mask.astype(np.int32) - np.array(color))
            distance = np.sum(diff, axis=-1)
            matches = distance < 10  # 容忍度：总差值小于10
            mask[matches] = class_id
        
        # 检查是否有未分类的像素
        unclassified = np.sum([np.all(rgb_mask == color, axis=-1) 
                                for color in self.color_map.keys()], axis=0) == 0
        # if np.any(unclassified):
        #     print("Warning: Found unclassified pixels in the mask.")
        return mask


    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('L')
        mask_rgb = Image.open(msk_path).convert('RGB')
        # 转换为类别mask
        mask = self.rgb_to_mask(mask_rgb)
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'training':
            img = resize_with_aspect_and_pad(img, (256, 256), is_mask=False, fill=0)
            mask = resize_with_aspect_and_pad(mask, (256, 256), is_mask=True, fill=0)
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            angle = torch.randint(-10, 11, (1,)).item()
            img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=0)
            
        else:
            img = resize_with_aspect_and_pad(img, (256, 256), is_mask=False, fill=0)
            mask = resize_with_aspect_and_pad(mask, (256, 256), is_mask=True, fill=0)

        img = TF.to_tensor(img)  # [1, H, W]
        img = img * 2.0 - 1.0

        mask = torch.from_numpy(np.array(mask)).long()  # 直接从numpy转换，保持原始值
    
        # print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        # print(f"Mask unique values: {torch.unique(mask)}")

        mask_onehot = F.one_hot(mask, num_classes=self.num_classes)  # [H, W, 3]
        mask_onehot = mask_onehot.permute(2, 0, 1).float()  # [3, H, W]
        # mask_onehot = mask_onehot * 2.0 - 1.0  # 转换到[-1, 1]

        return (img, mask_onehot, name)
    