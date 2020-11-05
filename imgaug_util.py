from matplotlib import pyplot as plt

import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)
store_path = r'D:/iqiyi/short_expose/PixAug/aug'


def read_xml_annotation(image_id):
    # members = tree.findall('object')
    in_file = open(image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()
    box_list = []
    objects = root.findall('object')
    for object in objects:
        bndbox = object.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)

        box_list.append((xmin, ymin, xmax, ymax))
    return box_list


def change_xml_annotation(xml_path, new_boxes):
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    objects = xmlroot.findall('object')
    for i in range(len(objects)):
        bndbox = objects[1].find('bndbox')
        points = objects[1].find('shape')[0]
        new_target = new_boxes[i]
        new_xmin = new_target[0]
        new_ymin = new_target[1]
        new_xmax = new_target[2]
        new_ymax = new_target[3]

        bndbox.find('xmin').text = str(new_xmin)
        bndbox.find('ymin').text = str(new_ymin)
        bndbox.find('xmax').text = str(new_xmax)
        bndbox.find('ymax').text = str(new_ymax)

        points[0].text = str(new_xmin)
        points[1].text = str(new_ymin)
        points[2].text = str(new_xmax)
        points[3].text = str(new_ymax)

    tree.write(xml_path[:-4] + "_aug" + '.xml')


if __name__ == "__main__":

    img_path = r'D:/iqiyi/short_expose/PixAug/413090037169_0_right.jpg'
    xml_path = r'D:/iqiyi/short_expose/PixAug/413090037169_0_right.xml'
    store_path = r'D:/iqiyi/short_expose/PixAug/aug'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bndbox = read_xml_annotation(xml_path)
    if len(bndbox) == 1:
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=bndbox[0][0], y1=bndbox[0][1], x2=bndbox[0][2], y2=bndbox[0][3])
        ], shape=img.shape)
    elif len(bndbox) == 2:
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=bndbox[0][0], y1=bndbox[0][1], x2=bndbox[0][2], y2=bndbox[0][3]),
            ia.BoundingBox(x1=bndbox[1][0], y1=bndbox[1][1], x2=bndbox[1][2], y2=bndbox[1][3])
        ], shape=img.shape)
    elif len(bndbox) == 3:
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=bndbox[0][0], y1=bndbox[0][1], x2=bndbox[0][2], y2=bndbox[0][3]),
            ia.BoundingBox(x1=bndbox[1][0], y1=bndbox[1][1], x2=bndbox[1][2], y2=bndbox[1][3]),
            ia.BoundingBox(x1=bndbox[2][0], y1=bndbox[2][1], x2=bndbox[2][2], y2=bndbox[2][3])
        ], shape=img.shape)
    elif len(bndbox) == 4:
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=bndbox[0][0], y1=bndbox[0][1], x2=bndbox[0][2], y2=bndbox[0][3]),
            ia.BoundingBox(x1=bndbox[1][0], y1=bndbox[1][1], x2=bndbox[1][2], y2=bndbox[1][3]),
            ia.BoundingBox(x1=bndbox[2][0], y1=bndbox[2][1], x2=bndbox[2][2], y2=bndbox[2][3]),
            ia.BoundingBox(x1=bndbox[3][0], y1=bndbox[3][1], x2=bndbox[3][2], y2=bndbox[3][3])
        ], shape=img.shape)
    else:
        print("undesired number of bounding boxes! at most 4 and at least 1!!")
        exit(0)
    seq_rotate = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 10, "y": 10},
            scale=(0.8, 0.95),
            rotate=(-10, 10)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
    seq_det = seq_rotate.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
    image_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB : (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
          )

    image_before = bbs.draw_on_image(img, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2)
    Image.fromarray(image_before).save(store_path + "/" + "before.jpg")
    Image.fromarray(image_after).save(store_path + "/" + 'after.jpg')

    new_bndbox_list = []
    for i in range(len(bbs_aug.bounding_boxes)):
        new_bndbox = []
        new_bndbox.append(int(bbs_aug.bounding_boxes[i].x1))
        new_bndbox.append(int(bbs_aug.bounding_boxes[i].y1))
        new_bndbox.append(int(bbs_aug.bounding_boxes[i].x2))
        new_bndbox.append(int(bbs_aug.bounding_boxes[i].y2))
        new_bndbox_list.append(new_bndbox)
    # 修改xml tree 并保存
    change_xml_annotation(xml_path, new_bndbox_list)

seq_test = iaa.Sequential([
    iaa.Crop(px=(0, 16)),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])
