import tqdm
import os

from pycocotools.coco import COCO
from pycocotools.mask import decode
import pycocotools.mask as maskUtils
import cv2
import sys
sys.path.append('..')


def annToRLE(segm, w, h):
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

split='val'
anno_file = '../data/coco/annotations/person_keypoints_{}2017.json'.format(split)
# ave_root = '/data1/images/{}2020_mask'.format(split)

'''
if not os.path.exists(save_root):
    os.makedirs(save_root, exist_ok=True)
'''

coco = COCO(anno_file)
imgIds = coco.getImgIds()
print('found {} imgs in {}'.format(len(imgIds), split))
# print(imgIds)

for imgId in tqdm.tqdm(imgIds):
    # print(coco.loadImgs(imgId))
    # [{'license': 2, 'file_name': '000000071451.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000071451.jpg', 'height': 640, 'width': 383, 'date_captured': '2013-11-21 00:12:55', 'flickr_url': 'http://farm8.staticflickr.com/7296/9116234211_a008c7d23c_z.jpg', 'id': 71451}]
    im_ann = coco.loadImgs(imgId)[0]
    width = im_ann['width']
    height = im_ann['height']

    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    objs = coco.loadAnns(ids=annIds)
    # print(objs)
    # generate mask
    for obj in objs:
        # print(obj)
        segs = obj['segmentation']
        rle = annToRLE(segm=segs, w=width, h=height)
        bimask = decode(rle)
        
        # filename = im_ann['file_name']
        # cv2.imwrite(os.path.join(save_root, filename).replace('jpg', 'png'), bimask * 255.)
        # cv2.imwrite('./test.png', bimask * 255)