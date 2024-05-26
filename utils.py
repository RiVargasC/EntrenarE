import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class ImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_img_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_img = cv2.imread(lr_img_path, cv2.IMREAD_COLOR)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.imread(hr_img_path, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        lr_img = self.to_tensor(lr_img)
        hr_img = self.to_tensor(hr_img)

        return lr_img, hr_img

def get_data_loaders(lr_dir, hr_dir, batch_size=16, num_workers=4):
    dataset = ImageDataset(lr_dir, hr_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
