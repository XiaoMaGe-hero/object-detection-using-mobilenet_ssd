# 图像800x640的串行排列
# 任务是逐次旋转bbox对应区域
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

    name_str = "_objrotate_" + str(i) + "_" + str(rotate_angle[0])
    tree.write(store_path + xml_path.split('\\')[-1][:-4] + name_str + '.xml')


def main(rotate_angle=[10, 10], gamma=0.1):
    path_list = []
    path_list_copy = []
    sourece_path = r'D:\iqiyi\DATA\organized2020-11-heri2ver'
    store_path = r'D:\iqiyi\DATA\organized2020-11-heri2ver_bbox_rotated'
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
                    # image.shape 800 640 3  转置也不行？ shape[0] ver, shape[1] heri, shape[3] channels
                    seq_rotate = iaa.Sequential([
                        # iaa.Flipud(0.5),  # vertically flip 20% of all images
                        # iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
                        iaa.Affine(
                            # translate_px={"x": 10, "y": 10},
                            # scale=(0.8, 0.95),
                            rotate=rotate_angle
                        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
                    ])
                    bndbox = read_xml_annotation(xml_name)
                    # 此处 坐标是【x，y】，而图片索引是【y，x】且y指向不明？？？
                    for i in range(len(bndbox)):
                        center_x = (bndbox[i][0] + bndbox[i][2]) // 2
                        center_y = (bndbox[i][1] + bndbox[i][3]) // 2
                        # 有些框出现在图片之外或者大部分在图片之外，很少，没有意义，直接略过
                        if center_y < 0 or center_x < 0:
                            continue
                        box_width = bndbox[i][2] - bndbox[i][0] + 1
                        box_height = bndbox[i][3] - bndbox[i][1] + 1
                        hypotenuse = int(np.sqrt(box_width**2 + box_height**2)) + 1
                        area_width = hypotenuse + int(box_width * gamma)
                        area_height = hypotenuse + int(box_height * gamma)

                        area_p_x = int(center_x - area_width / 2.0)
                        area_p_y = int(center_y - area_height / 2.0)

                        # area picture make
                        # 先把框中内容提取出来然后周围padding到期望的范围，不然可能会超出图片范围
                        area_img = np.zeros((area_width, area_height), dtype=np.uint8)
                        area_img = cv2.cvtColor(area_img, cv2.COLOR_GRAY2BGR)
                        # area_img [x, y]
                        box_area = img[bndbox[i][1]:bndbox[i][3], bndbox[i][0]:bndbox[i][2], :]
                        # box_area  [y, x] picture!!!!
                        box_area = box_area.transpose((1, 0, 2))
                        # box_area  [x, y]
                        box_in_area_x = int((area_width - box_width) // 2)
                        box_in_area_y = int((area_height - box_height) // 2)

                        print(box_area.shape)
                        print(box_in_area_x, box_in_area_y)
                        print(area_img.shape)
                        area_img[box_in_area_x: box_in_area_x + box_area.shape[0], box_in_area_y: box_in_area_y + box_area.shape[1], :] = box_area
                        # area done

                        area_img = area_img.transpose((1, 0, 2))

                        bbs = ia.BoundingBoxesOnImage([
                            ia.BoundingBox(x1=bndbox[i][0] - area_p_x, y1=bndbox[i][1] - area_p_y, x2=bndbox[i][2] - area_p_x, y2=bndbox[i][3] - area_p_y)
                        ], shape=area_img.shape)

                        # image_before = bbs.draw_on_image(area_img, size=1)
                        # Image.fromarray(image_before).save(
                        #     store_path + "/" + jpg_name.split('\\')[-1][:-4] + 'before.jpg')

                        seq_det = seq_rotate.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                        image_aug = seq_det.augment_images([area_img])[0]
                        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

                        # for test
                        image_after = bbs_aug.draw_on_image(image_aug, size=1)
                        # Image.fromarray(image_after).save(store_path + "/" + jpg_name.split('\\')[-1][:-4] + 'after.jpg')

                        # make up the final picture using image_aug, img, area_p_x, area_p_y
                        # img [y, x] image_aug[x, y]

                        # image_after = image_after.transpose((1, 0, 2))
                        print(jpg_name)

                        aa = area_p_y + int(image_after.shape[0])
                        bb = area_p_x + int(image_after.shape[1])
                        if area_p_x >= 0 and area_p_y >= 0:
                            if aa <= img.shape[0] and bb <= img.shape[1]:
                                img[area_p_y:aa, area_p_x:bb, :] = image_after
                            elif aa <= img.shape[0] and bb > img.shape[1]:
                                img[area_p_y:aa, area_p_x:, :] = image_after[:, :img.shape[1] - area_p_x, :]
                            elif aa > img.shape[0] and bb <= img.shape[1]:
                                img[area_p_y:, area_p_x:bb, :] = image_after[:img.shape[0] - area_p_y, :, :]
                            else:
                                img[area_p_y:, area_p_x:, :] = image_after[:img.shape[0] - area_p_y, :img.shape[1] - area_p_x, :]
                        elif area_p_x < 0 and area_p_y >= 0:
                            if aa <= img.shape[0] and bb <= img.shape[1]:

                                img[area_p_y:aa, 0:bb, :] = image_after[:, -area_p_x:, :]
                            elif aa <= img.shape[0] and bb > img.shape[1]:
                                img[area_p_y:aa, 0:, :] = image_after[:, -area_p_x:img.shape[1] - area_p_x, :]
                            elif aa > img.shape[0] and bb <= img.shape[1]:
                                img[area_p_y:, 0:bb, :] = image_after[:img.shape[0] - area_p_y, -area_p_x:, :]
                            else:
                                img[area_p_y:, 0:, :] = image_after[:img.shape[0] - area_p_y, -area_p_x:img.shape[1] - area_p_x, :]
                        elif area_p_x >= 0 and area_p_y < 0:
                            if aa <= img.shape[0] and bb <= img.shape[1]:
                                img[0:aa, area_p_x:bb, :] = image_after[-area_p_y:, :, :]
                            elif aa <= img.shape[0] and bb > img.shape[1]:
                                print(area_p_y, aa)
                                img[0:aa, area_p_x:, :] = image_after[-area_p_y:, :img.shape[1] - area_p_x, :]
                            elif aa > img.shape[0] and bb <= img.shape[1]:
                                img[0:, area_p_x:bb, :] = image_after[-area_p_y:img.shape[0] - area_p_y, :, :]
                            else:
                                img[0:, area_p_x:, :] = image_after[-area_p_y:img.shape[0] - area_p_y, :img.shape[1] - area_p_x, :]
                        else:
                            if aa <= img.shape[0] and bb <= img.shape[1]:
                                img[0:aa, 0:bb, :] = image_after[-area_p_y:, -area_p_x:, :]
                            elif aa <= img.shape[0] and bb > img.shape[1]:
                                img[0:aa, 0:, :] = image_after[-area_p_y:, -area_p_x:img.shape[1] - area_p_x, :]
                            elif aa > img.shape[0] and bb <= img.shape[1]:
                                img[0:, 0:bb, :] = image_after[-area_p_y:img.shape[0] - area_p_y, -area_p_x:, :]
                            else:
                                img[0:, 0:, :] = image_after[-area_p_y:img.shape[0] - area_p_y, -area_p_x:img.shape[1] - area_p_x, :]

                        name_str = "_objrotate_" + str(i) + "_" + str(rotate_angle[0])
                        Image.fromarray(img).save(normal_store + "\\" + jpg_name.split('\\')[-1][:-4] + name_str + ".jpg")


                        # check and fix the bbox that locates outside the pix
                        # bbs_aug.remove_out_of_image().clip_out_of_image()
                        img_wh = img.shape[:2]
                        new_bndbox = []
                        print(img_wh[0], img_wh[1])
                        new_bndbox.append(max(int(bbs_aug.bounding_boxes[0].x1 + area_p_x), 0))
                        new_bndbox.append(max(int(bbs_aug.bounding_boxes[0].y1 + area_p_y), 0))
                        new_bndbox.append(min(int(bbs_aug.bounding_boxes[0].x2 + area_p_x), img_wh[1]))
                        new_bndbox.append(min(int(bbs_aug.bounding_boxes[0].y2 + area_p_y), img_wh[0]))
                        # 修改xml tree 并保存

                        change_xml_annotation(xml_name, new_bndbox, i, img_wh, normal_store)
# 超参数 ： 外扩区域比例 gamma


gamma = 0.1
rotate_angle = [10, 10]
main(rotate_angle, gamma)
