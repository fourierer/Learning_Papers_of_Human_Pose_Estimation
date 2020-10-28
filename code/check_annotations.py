#import coco
#import pycocotools.coco as coco
import json


ann_val_file = './annotations/instances_val2014.json'
#coco_val = coco(ann_val_file)

with open(ann_val_file,'r',encoding='utf8')as fp:
    ann_data = json.load(fp)

print(type(ann_data))
print(type(ann_data['images']))
print(len(ann_data['categories']))
print(len(ann_data['images']))
print(len(ann_data['annotations']))

