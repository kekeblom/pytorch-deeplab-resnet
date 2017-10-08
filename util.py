import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_list, image_folder, gt_folder):
        self.image_list = image_list
        self.image_folder = image_folder
        self.gt_folder = gt_folder

    def __getitem__(self, index):
        filename = self.image_list[index]
        image = cv2.imread(os.path.join(
            self.image_folder, filename + '.jpg')).astype(float)
        ground_truth = cv2.imread(os.path.join(
            self.gt_folder, filename + '.png'))[:, :, 0]
        return image, ground_truth

    def __len__(self):
        return len(self.image_list)


