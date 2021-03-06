'''
Created using tutorial: 
- https://blog.francium.tech/object-detection-with-faster-rcnn-bc2e4295bf49
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Other folders are from:
- https://github.com/pytorch/vision/tree/master/references/detection
'''

# Import Libraries
import numpy as np
import cv2
import torch 
import torchvision 
from engine import train_one_epoch, evaluate
from data_reading import WindowDataset
from torch.utils.data import DataLoader

print(torchvision.__version__)

def get_model():
    # Initialize Model, from pytorch tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    num_classes = 2 # window + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features # Number of input features for the classifier
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'running on device {device}')
model = get_model()

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

train = DataLoader(WindowDataset(), batch_size=1)

num_epochs = 40

for epoch in range(num_epochs):
    # Train for one epoch, while printing every 10 iterations
    train_one_epoch(model, optimizer, train, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, train, device)

print("DONE")