import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, labels_folder = 'data/labels/', images_folder = 'data/images/'):
        super(Dataset, self).__init__()
        self.images, self.labels = [], []

        image_width = 128
        image_height = 96

        for folder in os.listdir(labels_folder):
            if folder != '.DS_Store':
                for filename in os.listdir(f"{labels_folder}{folder}"):
                    if filename.endswith("YOLO"):
                        for _file in os.listdir(f"{labels_folder}{folder}/{filename}"):
                            with open(f"{labels_folder}{folder}/{filename}/{_file}") as f:
                                lines = []
                                for line in f:
                                    line = line.split()
                                    label, rest = int(line[0]), line[1:]
                                    xmin, ymin, w, h = map(float, rest)
                                    xmin, ymin = int(xmin*image_width), int(ymin*image_height)
                                    w, h = int(w*image_width), int(h*image_height)
                                    lines.append([label, xmin, ymin, w+xmin, h+ymin])
                                lines = np.array(lines)
                                self.labels.append(torch.tensor(lines))
                            image = torch.tensor(cv2.cvtColor(cv2.imread(f"{images_folder}/{folder}/{_file[:-3]}jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                            image = image.permute(2,0,1).double()
                            print(image.shape)
                            self.images.append(image)
        
    def __getitem__(self, idx):
        boxes = self.labels[idx][:,1:]
        labels = self.labels[idx][:,0]
        print(boxes.shape)
        return (self.images[idx:idx+1], [{'boxes':boxes, 'labels':labels}])
    
    def __len__(self):
        return len(self.labels)
