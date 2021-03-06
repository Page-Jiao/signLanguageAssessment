import sys

import torch
import cv2
import logging
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.structures.instances import Instances
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor

torch.cuda.set_device(3)

def change_size(img):
    b=cv2.threshold(img,15,255,cv2.THRESH_BINARY)          #调整裁剪效果
    binary_image=b[1]               #二值图--具有三通道
    binary_image=cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)       #改为单通道
 
    x=binary_image.shape[0]
    y=binary_image.shape[1]
    edges_x=[]
    edges_y=[]
    for i in range(x):
        for j in range(y):
            if binary_image[i][j]==255:
             edges_x.append(i)
             edges_y.append(j)
 
    left=min(edges_x)               #左边界
    right=max(edges_x)              #右边界
    width=right-left                #宽度
    bottom=min(edges_y)             #底部
    top=max(edges_y)                #顶部
    height=top-bottom               #高度
 
    pre1_picture=img[left:left+width,bottom:bottom+height]        #图片截取
    return pre1_picture                                             #返回图片数据

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file("~/detectron2-master/projects/Densepose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")


cfg.MODEL.WEIGHTS = "~/detectron2-master/projects/Densepose/model_final_162be9.pkl"

cfg.freeze()
predictor = DefaultPredictor(cfg)

img = cv2.imread("~/signLanguageAssesment/data/1.jpg")

print(img.shape)

outputs = predictor(img)["instances"]

scores = outputs.get("scores")
box = outputs.get("pred_boxes").tensor.cpu().numpy()[0]

x1 = int(box[1])
y1 = int(box[0])
x2 = int(box[3])
y2 = int(box[2])

print(box.shape)
print(x1,y1,x2,y2)

mask_lefthand = np.zeros(img.shape[0:2], dtype=np.uint8)
mask_righthand = np.zeros(img.shape[0:2], dtype=np.uint8)

print(mask_lefthand.shape)
print(mask_lefthand)
result, _ = DensePoseResultExtractor()(outputs)

print(scores.size())
print(scores)
labels = result[0].labels.cpu().numpy()

for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if labels[i,j] == 3:
            mask_righthand[i+x1,j+y1] = 255
        elif labels[i,j] == 4:
            mask_lefthand[i+x1,j+y1] = 255
# uv = result[0].uv.cpu().numpy()
# print(labels.shape)
# print(uv.shape)

image_left_hand = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask = mask_lefthand)
image_right_hand = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask = mask_righthand)

image_left_hand = change_size(image_left_hand)
image_right_hand = change_size(image_right_hand)

cv2.imwrite("~/signLanguageAssesment/result/1_left.jpg", image_left_hand)
cv2.imwrite("~/signLanguageAssesment/result/1_right.jpg", image_right_hand)