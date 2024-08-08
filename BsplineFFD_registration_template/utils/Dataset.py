import os
import torch
import numpy as np
import csv
import nibabel as nib
from torch.utils.data import Dataset

class Tongji_Dataset_train(Dataset):
    def __init__(self, csv_file, crop_size=128):
        self.data = []
        self.crop_size = crop_size

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                folder_path = row['folder'].strip()
                file_path = os.path.join(folder_path, f"{row['filetype']}_{row['filepairs']}.nii.gz")
                '''
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                h, w = data.shape[:2]  # 获取未裁剪图像的高度和宽度
                crop_start_h = (h - crop_size) // 2  # 计算裁剪起始点的高度
                crop_start_w = (w - crop_size) // 2  # 计算裁剪起始点的高度
                cropped_data = data[crop_start_h:crop_start_h+self.crop_size, crop_start_w:crop_start_w+self.crop_size,:]
                cropped_data = (cropped_data - cropped_data.min()) / (cropped_data.max() - cropped_data.min())
                '''
                if 'Cine' in row['filetype']:
                    self.data.append({'img_moving': file_path})
                elif 'DE' in row['filetype']:
                    self.data[-1]['img_fixed'] = file_path


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Tongji_Dataset_valid(Dataset):
    def __init__(self, csv_file, crop_size=128):
        self.data = []
        self.crop_size = crop_size

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                folder_path = row['folder'].strip()
                file_path = os.path.join(folder_path, f"{row['filetype']}_{row['filepairs']}.nii.gz")
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                h, w = data.shape[:2]  # 获取未裁剪图像的高度和宽度
                crop_start_h = (h - crop_size) // 2  # 计算裁剪起始点的高度
                crop_start_w = (w - crop_size) // 2  # 计算裁剪起始点的高度
                cropped_data = data[crop_start_h:crop_start_h+self.crop_size, crop_start_w:crop_start_w+self.crop_size,:]
                cropped_data = (cropped_data - cropped_data.min()) / (cropped_data.max() - cropped_data.min())

                if 'Cine' in row['filetype']:
                    self.data.append({'img_moving': cropped_data})
                elif 'DE' in row['filetype']:
                    self.data[-1]['img_fixed'] = cropped_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
class Tongji_Dataset_test(Dataset):
    def __init__(self, csv_file, crop_size=128):
        self.data = []
        self.crop_size = crop_size

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                folder_path = row['folder'].strip()
                file_path = os.path.join(folder_path, f"{row['filetype']}_{row['filepairs']}.nii.gz")
                if 'Cine' in row['filetype']:
                    self.data.append({'img_moving':file_path})
                elif 'DE' in row['filetype']:
                    self.data[-1]['img_fixed'] = file_path


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    

