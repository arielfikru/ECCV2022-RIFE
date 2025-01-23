import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        
        # Use absolute path
        self.data_root = os.path.abspath('/content/drive/MyDrive/vimeo_triplet')
        self.image_root = os.path.join(self.data_root, 'sequences')
        
        print(f"Data root path: {self.data_root}")
        print(f"Image root path: {self.image_root}")
        
        # Check if directories exist
        if not os.path.exists(self.data_root):
            raise RuntimeError(f"Data root '{self.data_root}' does not exist")
        if not os.path.exists(self.image_root):
            raise RuntimeError(f"Image root '{self.image_root}' does not exist")
            
        # Load train/test lists with absolute paths
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        
        if not os.path.exists(train_fn):
            raise RuntimeError(f"Training list file '{train_fn}' not found")
        if not os.path.exists(test_fn):
            raise RuntimeError(f"Test list file '{test_fn}' not found")
            
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
            
        self.load_data()
        print(f"Dataset initialized with {len(self.meta_data)} samples")

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def getimg(self, index):
        # Use absolute path for image loading
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [
            os.path.join(imgpath, 'im1.png'),
            os.path.join(imgpath, 'im2.png'),
            os.path.join(imgpath, 'im3.png')
        ]
        
        # Check if files exist
        for path in imgpaths:
            if not os.path.exists(path):
                raise RuntimeError(f"Image file not found: {path}")

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        
        if img0 is None or gt is None or img1 is None:
            raise RuntimeError(f"Failed to load images from {imgpath}")
            
        timestep = 0.5
        return img0, gt, img1, timestep

    def __len__(self):
        return len(self.meta_data)

    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def __getitem__(self, index):
        try:
            img0, gt, img1, timestep = self.getimg(index)
            if self.dataset_name == 'train':
                img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, :, ::-1]
                    img1 = img1[:, :, ::-1]
                    gt = gt[:, :, ::-1]
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[::-1]
                    img1 = img1[::-1]
                    gt = gt[::-1]
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, ::-1]
                    img1 = img1[:, ::-1]
                    gt = gt[:, ::-1]
                if random.uniform(0, 1) < 0.5:
                    tmp = img1
                    img1 = img0
                    img0 = tmp
                    timestep = 1 - timestep
                # random rotation
                p = random.uniform(0, 1)
                if p < 0.25:
                    img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                    gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
                elif p < 0.5:
                    img0 = cv2.rotate(img0, cv2.ROTATE_180)
                    gt = cv2.rotate(gt, cv2.ROTATE_180)
                    img1 = cv2.rotate(img1, cv2.ROTATE_180)
                elif p < 0.75:
                    img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            timestep = torch.tensor(timestep).reshape(1, 1, 1)
            return torch.cat((img0, img1, gt), 0), timestep
            
        except Exception as e:
            print(f"Error loading sample {index}: {str(e)}")
            raise
