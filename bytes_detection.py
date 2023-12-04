import sys
import os
from dotenv import load_dotenv
from PIL import Image
import csv
import torch
import torch.nn as nn
import transforms as T
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import requests as re
import json

load_dotenv()

'''
This code takes in a list of images as paths, and will run a trained object detection model on each of these images. 
From these images, it will find and locate the respective labels/items. It will then use these detected objects to
make a request to EdamamAPI's recipe catalog, where we'll be given a list of possible recipes and links to making them.
'''

# globals
path = 'model.pth'
confidence = 0.7
APP_ID = os.environ.get('APP_ID')
APP_KEY = os.environ.get('APP_KEY')

def get_model(num_classes):
  weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features

  # replace pretrained head w/ new head
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

def analyze_image(image):
    np_sample_image = np.array(image.convert("RGB"))

    transformed_img = torchvision.transforms.transforms.ToTensor()(
        torchvision.transforms.ToPILImage()(np_sample_image))

    pred = model([transformed_img])
    pred_labels = [idx_to_class[i - 1] for i in pred[0]['labels'].numpy()]
    pred_score = list(pred[0]['scores'].detach().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]

    return [pred_labels[ind] for ind in pred_t]


def get_classes():
    classes = []
    with open('oidv7-class-descriptions-boxable-modified.csv') as classes_file:
        reader_obj = csv.reader(classes_file)
        for index, line in enumerate(reader_obj):
            if index == 0:
                continue

            classes.append(line[1])

    return classes


classes = get_classes()
idx_to_class = {i: j for i, j in enumerate(classes)}
model = get_model(len(classes)+2)
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()

detected_classes = set()
# passed in through command line
if len(sys.argv) > 1:
    for path in sys.argv[1:]:
        image = Image.open(path)
        detected = analyze_image(image)

        for item in detected:
            detected_classes.add(item)

# passed in through input file
else:
    for path in sys.stdin:
        image = Image.open(path.strip())
        detected = analyze_image(image)

        for item in detected:
            detected_classes.add(item)

print(detected_classes)

url = f'https://api.edamam.com/search?q={",".join(detected_classes)}&app_id={APP_ID}&app_key={APP_KEY}'
response = re.get(url)

data = response.json()
for recipe in data['hits']:
    print(recipe['recipe']['label'])
    print(recipe['recipe']['url'])







