import os
import cv2
import torchvision
import numpy as np
from torch.utils.data import Dataset
import torch

class CityScapeTestDataset(Dataset):
    def __init__(self):
        super(CityScapeTestDataset,self).__init__()
        self._root_path = os.path.join('/datasets', 'cityscapes/')
        self._image_path = os.path.join(self._root_path, 'leftImg8bit')
        self._gt_path = os.path.join(self._root_path, 'gtFine')
        self._test_img_path = os.path.join(self._image_path, 'test')
        self._test_gt_path = os.path.join(self._gt_path, 'test')
        self._img_names_test = []
        self._gt_names_test = []
        berlines = os.listdir(self._test_img_path)
        for berlin in berlines:
            for file in os.listdir(os.path.join(self._test_img_path, berlin)):
                if file.endswith('.png'):
                    self._img_names_test.append(os.path.join(berlin,file))
            for file in os.listdir(os.path.join(self._test_gt_path, berlin)):
                if file.endswith('labelIds.png'):
                    self._gt_names_test.append(os.path.join(berlin,file))
        self._img_names_test.sort()
        self._gt_names_test.sort()

    def __getitem__(self, index):
        img_path = os.path.join(self._test_img_path, self._img_names_test[index])
        img = np.array(cv2.imread(img_path))
        gt_path = os.path.join(self._test_gt_path, self._gt_names_test[index])
        gt = np.array(cv2.imread(gt_path,0))
        img = torchvision.transforms.functional.to_tensor(img)
        gt = torchvision.transforms.functional.to_tensor(gt)
        gt = gt*256
        return img,gt.int()
        
    def __len__(self):
        return len(self._img_names_test)
