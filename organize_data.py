import os
import torch
import cv2
import json

image_width = 128
image_height = 96
labels_folder = 'data/test/labels/'
images_folder = 'data/test/images/'

output = {'categories':[{'id':1, 'name':'window'}], 'images':[], 'annotations':[]}
count = 1
img_count = 0
for folder in os.listdir(labels_folder):
    if folder != '.DS_Store':
        for filename in os.listdir(f"{labels_folder}{folder}"):
            if filename.endswith("YOLO"):
                for i, _file in enumerate(os.listdir(f"{labels_folder}{folder}/{filename}")):
                    img_count += 1
                    with open(f"{labels_folder}{folder}/{filename}/{_file}") as f:
                        boxes = []
                        labels = []
                        for line in f:
                            line = line.split()
                            label, rest = int(line[0]), line[1:]
                            xcenter, ycenter, w, h = map(float, rest)
                            xmin, ymin = int(xcenter*image_width-w*image_width/2), int(ycenter*image_height-h*image_height/2)
                            w, h = int(w*image_width), int(h*image_height)
                            if w<2 or h<2 or image_width-xmin < 2 or image_height-ymin < 2:
                                continue
                            output['annotations'].append({})
                            box = [xmin, ymin, w, h]
                            output['annotations'][-1] = {'id':count, 'image_id':img_count, 'category_id':1, 'bbox':box, 'iscrowd':0, 'area':(box[2]-box[0])*(box[3]-box[1])}
                            count+=1
                    output['images'].append({})
                    output['images'][-1] = {'id':img_count,'file_name':f'{folder}/{_file[:-3]}jpg'}

with open('data/test/anno','w') as f:
    json.dump(output,f)