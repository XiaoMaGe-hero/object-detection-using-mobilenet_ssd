# 图像800x640的串行排列
# 任务是逐次以bbox为中心旋转整张图片
# 处理jpg 和 xml文件
# 按照源目录和标准命名备份

import xml.etree.ElementTree as ET
import os
import numpy as np
from os import getcwd
from PIL import Image
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
ia.seed(1)


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


def change_xml_annotation(xml_path, new_boxes, i, image_size, store_path):
    # i the sequence number of this box
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()

    xmlroot.find('size')[0].text = str(image_size[1])    # width
    xmlroot.find('size')[1].text = str(image_size[0])    # height

    objects = xmlroot.findall('object')
    bndbox = objects[i].find('bndbox')
    points = objects[i].find('shape')[0]

    bndbox.find('xmin').text = str(new_boxes[0])
    bndbox.find('ymin').text = str(new_boxes[1])
    bndbox.find('xmax').text = str(new_boxes[2])
    bndbox.find('ymax').text = str(new_boxes[3])

    points[0].text = str(new_boxes[0])
    points[1].text = str(new_boxes[1])
    points[2].text = str(new_boxes[2])
    points[3].text = str(new_boxes[3])

    name_str = "_imgrotate_" + str(i) + "_" + str(rotate_angle[0])
    tree.write(store_path + xml_path.split('\\')[-1][:-4] + name_str + '.xml')


def main(rotate_angle=10, scale=1.0):
    path_list = []
    path_list_copy = []
    sourece_path = r'D:\iqiyi\DATA\organized2020-11-heri2ver'
    store_path = r'D:\iqiyi\DATA\organized2020-11-heri2ver-imgrotated'
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    for i in os.listdir(sourece_path):
        temp_path = sourece_path + "\\" + i
        path_list.append(temp_path)   # 保存绝对地址
        path_list_copy.append(store_path + "\\" + i)
        if not os.path.isdir(store_path + "\\" + i):
            os.mkdir(store_path + "\\" + i)
    while len(path_list):
        temp = path_list[-1]
        temp_copy = path_list_copy[-1]
        path_list.pop(-1)
        path_list_copy.pop(-1)
        file_list = os.listdir(temp)
        for i in file_list:
            if os.path.isdir(temp + "\\" + i):
                path_list.append(temp + "\\" + i)
                path_list_copy.append(temp_copy + "\\" + i)
                os.mkdir(temp_copy + "\\" + i)
            else:
                if i[-4:] == r'.xml':
                    jpg_name = temp + "\\" + i[:-4] + r'.jpg'
                    xml_name = jpg_name[:-4] + r'.xml'
                    print(jpg_name)
                    normal_store = temp_copy + "\\"

                    img = cv2.imread(jpg_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #  hape[0] ver, shape[1] heri, shape[3] channels
                    (h, w) = img.shape[:2]

                    boxes = read_xml_annotation(xml_name)
                    for j in range(len(boxes)):
                        x0 = boxes[j][0]
                        x1 = boxes[j][2]
                        y0 = boxes[j][1]
                        y1 = boxes[j][3]

                        center = ((x0 + x1) // 2, (y0 + y1) // 2)
                        if center[0] < 0 or center[1] < 0:
                            continue
                        M = cv2.getRotationMatrix2D(center, rotate_angle[0], scale)
                        sinv = abs(M[0][1])
                        cosv = abs(M[0][0])
                        img_r = cv2.warpAffine(img, M, (w, h))

                        a = center[0]
                        b = center[1]
                        xmin = min(cosv * (x0 - a) + sinv * (y0 - b), cosv * (x0 - a) + sinv * (y1 - b),
                                   cosv * (x1 - a) + sinv * (y0 - b), cosv * (x1 - a) + sinv * (y1 - b)) + a
                        xmax = max(cosv * (x0 - a) + sinv * (y0 - b), cosv * (x0 - a) + sinv * (y1 - b),
                                   cosv * (x1 - a) + sinv * (y0 - b), cosv * (x1 - a) + sinv * (y1 - b)) + a
                        ymin = min(-sinv * (x0 - a) + cosv * (y0 - b), -sinv * (x0 - a) + cosv * (y1 - b),
                                   -sinv * (x1 - a) + cosv * (y0 - b), -sinv * (x1 - a) + cosv * (y1 - b)) + b
                        ymax = max(-sinv * (x0 - a) + cosv * (y0 - b), -sinv * (x0 - a) + cosv * (y1 - b),
                                   -sinv * (x1 - a) + cosv * (y0 - b), -sinv * (x1 - a) + cosv * (y1 - b)) + b

                        # bbs_prediction = ia.BoundingBoxesOnImage([
                        #     ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
                        # ], shape=img.shape)
                        # img_r = bbs_prediction.draw_on_image(img_r, size=1, color=111)

                        name_str = "_imgrotate_" + str(j) + "_" + str(rotate_angle[0])
                        cv2.imwrite(normal_store + "\\" + jpg_name.split('\\')[-1][:-4] + name_str + ".jpg", img_r)
                        img_wh = img.shape[:2]
                        new_bndbox = []
                        new_bndbox.append(max(xmin, 0))
                        new_bndbox.append(max(ymin, 0))
                        new_bndbox.append(min(xmax, img_wh[1]))
                        new_bndbox.append(min(ymax, img_wh[0]))
                        # 修改xml tree 并保存
                        change_xml_annotation(xml_name, new_bndbox, j, img_wh, normal_store)
# 超参数 ： 外扩区域比例 gamma


rotate_angle = [10, 10]
scale = 1.0
main(rotate_angle, scale)