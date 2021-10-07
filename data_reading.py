import numpy as np
import os
import cv2
import torch
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import ToTensor

class WindowDataset(CocoDetection):
    def __init__(self, images_folder = 'data/images/', labels_path = 'data/anno'):
        super().__init__(images_folder, labels_path)
        
        
    def __getitem__(self, idx):
        img,labs = super().__getitem__(idx)
        T = ToTensor()
        img = T(img)
        if len(labs) == 0:
            return None, None
        nTargets = {'boxes':[], 'labels':[], 'image_id':torch.tensor(labs[0]['image_id'],dtype=torch.int64), 'area':torch.tensor(labs[0]['image_id'],dtype=torch.int64)}
        for lab in labs:
            box = lab['bbox'].copy()
            box[2] += box[0]
            box[3] += box[1]
            if box[2]-box[0] < 1:
                continue
            if box[3]-box[1] < 1:
                continue
            nTargets['boxes'].append(box)
            nTargets['labels'].append(lab['category_id'])
        nTargets['boxes'] = torch.tensor(nTargets['boxes']).float()
        nTargets['labels'] = torch.tensor(nTargets['labels'],dtype=torch.int64)
        return img.double(), nTargets

    
    def __len__(self):
        return super().__len__()


'''
self.data = []
        image_width = 128
        image_height = 96

        for folder in os.listdir(labels_folder):
            if folder != '.DS_Store':
                for filename in os.listdir(f"{labels_folder}{folder}"):
                    if filename.endswith("YOLO"):
                        for _file in os.listdir(f"{labels_folder}{folder}/{filename}"):
                            self.data.append([None,None])
                            with open(f"{labels_folder}{folder}/{filename}/{_file}") as f:
                                boxes = []
                                labels = []
                                for line in f:
                                    line = line.split()
                                    label, rest = int(line[0]), line[1:]
                                    xmin, ymin, w, h = map(float, rest)
                                    xmin, ymin = int(xmin*image_width), int(ymin*image_height)
                                    w, h = int(w*image_width), int(h*image_height)
                                    boxes.append(torch.tensor([xmin, ymin, min(w+xmin, image_width-1), min(h+ymin, image_height-1)]))
                                    labels.append(label)
                                self.data[-1][1] = {'boxes':torch.stack(boxes).float(), 'labels':torch.tensor(labels).long()+1}
                            image = torch.tensor(cv2.cvtColor(cv2.imread(f"{images_folder}/{folder}/{_file[:-3]}jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                            image = image.permute(2,0,1).double()
                            self.data[-1][0] = image
'''