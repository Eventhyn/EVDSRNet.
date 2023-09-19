from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import time
import os
from basicsr.utils import USMSharp
import numpy as np
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
aug = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5)
    #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01)
)
class Video_Provider_davis(Dataset):
    def __init__(self, base_path, txt_file, im_size=256, frames=3,aug=False):
        super(Video_Provider_davis, self).__init__()
        self.base_path = base_path
        self.txt_file = open(txt_file, 'r').readlines()
        self.im_size = im_size
        self.trans = transforms.ToTensor()
        self.frames = frames
        self.aug = aug
    def _get_file_name(self, index):
        """
        Read consecutive frames within index-th data starting at the start-th frame
        :param index: number of video in dataset
        :return:
        """
        res = []
        start = 0
        for i in range(self.frames):
            res.append(os.path.join(self.base_path, self.txt_file[index].strip(), '%05d.jpg'%i))
        return res

    @staticmethod
    def _get_random_sigma():
        #r = np.random.randint(5,15)
        r = 20.0/255
        return r
        
    def _get_crop_h_w(self):
        h = np.random.randint(0, 256 - self.im_size + 1)
        w = np.random.randint(0, 256 - self.im_size + 1)
        return h, w

    def __getitem__(self, index):
        img_files = self._get_file_name(index)
        hs, ws = self._get_crop_h_w()
        sigma = self._get_random_sigma()
        HQ = torch.zeros(self.frames, 3, self.im_size, self.im_size)
        LQ = torch.zeros(self.frames, 3, self.im_size, self.im_size)
        for i, file in enumerate(img_files):
            img = Image.open(file)
            img = self.trans(img)[:, hs:hs+self.im_size, ws:ws+self.im_size]
            HQ[i,...] = img
        if self.aug == True:
            HQ = aug(HQ)
        LR = torch.nn.functional.interpolate(HQ, scale_factor=0.25, mode='bicubic', antialias=True)
        noise = sigma * torch.randn_like(LR)
        LQ = LR + noise
        
        return LQ,LR,HQ

    def __len__(self):
        return len(self.txt_file)

class Video_Provider_val(Dataset):
    def __init__(self, base_path, txt_file, im_size=256, frames=3):
        super(Video_Provider_val, self).__init__()
        self.base_path = base_path
        self.txt_file = open(txt_file, 'r').readlines()
        self.im_size = im_size
        self.trans = transforms.ToTensor()
        self.frames = frames

    def _get_file_name(self, index):
        """
        Read consecutive frames within index-th data starting at the start-th frame
        :param index: number of video in dataset
        :return:
        """
        res = []
        start = 0
        for i in range(1,8):
            res.append(os.path.join(self.base_path, self.txt_file[index].strip(), 'im{}.png'.format(i)))
        return res

    @staticmethod
    def _get_random_sigma():
        r = 20.0
        r = r/255
        return r
        
    def _get_crop_h_w(self):
        h = np.random.randint(0, 256 - self.im_size + 1)
        w = np.random.randint(0, 448 - self.im_size + 1)
        return h, w

    def __getitem__(self, index):
        img_files = self._get_file_name(index)
        sigma = self._get_random_sigma()
        HQ = torch.zeros(self.frames, 3, 256, 448)
        LQ = torch.zeros(self.frames, 3, 256, 448)
        for i, file in enumerate(img_files):
            img = Image.open(file)
            img = self.trans(img)
            HQ[i,...] = img
        LR = torch.nn.functional.interpolate(HQ, scale_factor=0.25, mode='bicubic', antialias=True)
        noise = sigma * torch.randn_like(LR)
        LQ = LR + noise
        return LQ,LR,HQ
    def __len__(self):
        return len(self.txt_file)
  
class Video_Provider_val_udm(Dataset):
    def __init__(self, base_path, txt_file, im_size=256, frames=3):
        super(Video_Provider_val_udm, self).__init__()
        self.base_path = base_path
        self.txt_file = open(txt_file, 'r').readlines()
        self.im_size = im_size
        self.trans = transforms.ToTensor()
        self.frames = frames

    def _get_file_name(self, index):
        """
        Read consecutive frames within index-th data starting at the start-th frame
        :param index: number of video in dataset
        :return:
        """
        res = []
        for i in range(32):
            res.append(os.path.join(self.base_path, self.txt_file[index].strip(), '%04d.png'%i))
        return res

    @staticmethod
    def _get_random_sigma():
        r = 20.0
        r = r/255
        return r

    def __getitem__(self, index):
        img_files = self._get_file_name(index)
        sigma = self._get_random_sigma()
        HQ = torch.zeros(self.frames, 3, 704, 1248)
        LQ = torch.zeros(self.frames, 3, 704, 1248)
        for i, file in enumerate(img_files):
            img = Image.open(file)
            img = self.trans(img)[:, 0:704, 0:1248]
            HQ[i,...] = img
        LR = torch.nn.functional.interpolate(HQ, scale_factor=0.25, mode='bicubic', antialias=True)
        noise = sigma * torch.randn_like(LR)
        LQ = LR + noise
        return LQ,LR,HQ

    def __len__(self):
        return len(self.txt_file)