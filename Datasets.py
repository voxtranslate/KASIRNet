import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os

# ==========================================
# 9. Dataset & Training Pipeline
# ==========================================
class UniversalRestorationDataset(Dataset):
    def __init__(self, data_config, patch_size=256, is_train=True):
        self.samples = []
        self.patch_size = patch_size
        self.is_train = is_train
        self.task_to_id = {name: i for i, name in enumerate(data_config.keys())}
        for task_name, paths in data_config.items():
            input_dir = Path(paths['input'])
            gt_dir = Path(paths['gt'])
            input_files = sorted(list(input_dir.glob('*.*')))[:2500]
            gt_files = sorted(list(gt_dir.glob('*.*')))[:2500]
            if len(input_files) > 0:
                for inp, gt in zip(input_files, gt_files):
                    self.samples.append({'input_path': str(inp), 'gt_path': str(gt), 'task_name': task_name, 'task_id': self.task_to_id[task_name]})

    def __len__(self):
        return len(self.samples)

    def transform(self, image, gt):
        w, h = image.size
        if w < self.patch_size or h < self.patch_size:
            new_w = max(w, self.patch_size)
            new_h = max(h, self.patch_size)
            image = TF.resize(image, [new_h, new_w])
            gt    = TF.resize(gt,    [new_h, new_w])
            w, h  = image.size

        if self.is_train:
            left = random.randint(0, w - self.patch_size)
            top  = random.randint(0, h - self.patch_size)
            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            gt    = gt.crop((left, top, left + self.patch_size, top + self.patch_size))
            if random.random() > 0.5:
                image, gt = TF.hflip(image), TF.hflip(gt)
            if random.random() > 0.5:
                image, gt = TF.vflip(image), TF.vflip(gt)
        else:
            left = (w - self.patch_size) // 2
            top  = (h - self.patch_size) // 2
            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            gt    = gt.crop((left, top, left + self.patch_size, top + self.patch_size))

        return TF.to_tensor(image), TF.to_tensor(gt)

    def __getitem__(self, idx):
        item = self.samples[idx]
        inp_img = Image.open(item['input_path']).convert('RGB')
        gt_img = Image.open(item['gt_path']).convert('RGB')
        inp_tensor, gt_tensor = self.transform(inp_img, gt_img)
        return {'input': inp_tensor, 'gt': gt_tensor, 'task_name': item['task_name'], 'task_id': item['task_id']}
