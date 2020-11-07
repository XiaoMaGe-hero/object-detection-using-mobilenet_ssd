# 图像800x640的串行排列
# 任务是整图旋转，并相应旋转标注框
# 旋转后保证框不出界
# 处理jpg 和 xml文件
# 按照源目录和标准命名备份

import xml.etree.ElementTree as ET
import os
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


def change_xml_annotation(xml_path, new_boxes, image_size, store_path):
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()

    xmlroot.find('size')[0].text = str(image_size[1])    # width
    xmlroot.find('size')[1].text = str(image_size[0])    # height

    objects = xmlroot.findall('object')
    for i in range(len(objects)):
        bndbox = objects[i].find('bndbox')
        points = objects[i].find('shape')[0]
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
    for object in objects:
        bndbox = objects[i].find('bndbox')
        xmax = bndbox.find('xmin').text
        ymax = bndbox.find('ymin').text
        if int(xmax) >= image_size[1] or int(ymax) >= image_size[0]:
            xmlroot.remove(object)
    tree.write(store_path + xml_path.split('\\')[-1][:-4] + "_imgrotate_img_20" + '.xml')


def main(rotate_angle=[10, 10]):
    path_list = []
    path_list_copy = []
    sourece_path = r'D:\iqiyi\DATA\organized2020-11-heri2ver'
    store_path = r'D:\iqiyi\DATA\organized2020-11-heri2ver-rotated'
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

                    bndbox = read_xml_annotation(xml_name)
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
                        # we assume that there are at most four boxes
                        print("undesired number of bounding boxes! at most 4 and at least 1!!")
                        exit(0)
                    seq_rotate = iaa.Sequential([
                        iaa.Flipud(0.5),  # vertically flip 20% of all images
                        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
                        iaa.Affine(
                            # translate_px={"x": 10, "y": 10},
                            # scale=(0.8, 0.95),
                            rotate=rotate_angle
                        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
                    ])
                    seq_det = seq_rotate.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                    image_aug = seq_det.augment_images([img])[0]
                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

                    # for test
                    # for i in range(len(bbs.bounding_boxes)):
                    #     before = bbs.bounding_boxes[i]
                    #     after = bbs_aug.bounding_boxes[i]
                    #     print("BB : (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                    #         before.x1, before.y1, before.x2, before.y2,
                    #         after.x1, after.y1, after.x2, after.y2)
                    #           )
                    # image_before = bbs.draw_on_image(img, size=2)
                    # image_after = bbs_aug.draw_on_image(image_aug, size=2)
                    # Image.fromarray(image_before).save(store_path + "/" + jpg_name.split('\\')[-1][:-4] + "before.jpg")
                    # Image.fromarray(image_after).save(store_path + "/" + jpg_name.split('\\')[-1][:-4] + 'after.jpg')

                    Image.fromarray(image_aug).save(normal_store + jpg_name.split('\\')[-1][:-4] + "_imgrotate_img_20" + ".jpg")

                    # check and fix the bbox that locates outside the pix
                    # bbs_aug.remove_out_of_image().clip_out_of_image()
                    img_wh = image_aug.shape[:2]
                    print(img_wh[0], img_wh[1])
                    # img_wh = [height, width]
                    new_bndbox_list = []
                    for i in range(len(bbs_aug.bounding_boxes)):
                        new_bndbox = []
                        new_bndbox.append(int(bbs_aug.bounding_boxes[i].x1))
                        new_bndbox.append(int(bbs_aug.bounding_boxes[i].y1))
                        new_bndbox.append(min(int(bbs_aug.bounding_boxes[i].x2), img_wh[1]))
                        new_bndbox.append(min(int(bbs_aug.bounding_boxes[i].y2), img_wh[0]))
                        new_bndbox_list.append(new_bndbox)
                    # 修改xml tree 并保存
                    change_xml_annotation(xml_name, new_bndbox_list, img_wh, normal_store)


main(rotate_angle=[20, 20])
