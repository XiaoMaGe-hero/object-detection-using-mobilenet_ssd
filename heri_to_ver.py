# 原图像是400x1280的并行排列双图，现在要把他们变成800x640的串行排列
# 分割、拼接图像
# 处理xml文件

import cv2
from matplotlib import pyplot as plt
import os
import xml.etree.ElementTree as ET
import glob
import numpy as np


def tran_pix(path, flag=0):
    #  flag = 0, common pix
    #  flag = 1, box with error, store it in another place

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    print(h, w)
    assert h == 400
    assert w == 1280
    img_l = img[:, :640, :]
    img_r = img[:, 640:, :]
    img_tran = np.append(img_r, img_l, axis=0)
    if flag:
        cv2.imwrite(path[:-4] + '_t' + '.jpg', img_tran)  # a predefined address
    else:
        cv2.imwrite(path[:-4] + '_t' + '.jpg', img_tran)


def tran_xml(path):
    # revise : size
    # the first blob : move up 400
    # the second blob : move left 640
    xml = path[:-4] + '.xml'
    tree = ET.parse(xml)
    flag = 0

    tree.find('size')[0].text = str(640)
    tree.find('size')[1].text = str(800)
    # 默认只有两个标记框
    members = tree.findall('object')
    if len(members) == 1:
        if int(members[0].find('bndbox')[0].text) >= 640:
            # only a right box
            members[0].find('bndbox')[0].text = str(int(members[0].find('bndbox')[0].text) - 640)
            members[0].find('shape')[0][0].text = str(int(members[0].find('shape')[0][0].text) - 640)
            members[0].find('bndbox')[2].text = str(int(members[0].find('bndbox')[2].text) - 640)
            members[0].find('shape')[0][2].text = str(int(members[0].find('shape')[0][2].text) - 640)
        elif int(members[0].find('bndbox')[2].text) <= 640:
            # only a left box
            members[0].find('bndbox')[1].text = str(400+int(members[0].find('bndbox')[0].text))
            members[0].find('shape')[0][1].text = str(400+int(members[0].find('shape')[0][0].text))
            members[0].find('bndbox')[3].text = str(400+int(members[0].find('bndbox')[2].text))
            members[0].find('shape')[0][3].text = str(400+int(members[0].find('shape')[0][2].text))
        elif (int(members[0].find('bndbox')[2].text) + int(members[0].find('bndbox')[0].text)) / 2 <= 640:
            # mostly a left box
            members[0].find('bndbox')[2].text = str(640)
            members[0].find('shape')[0][2].text = str(640)

            members[0].find('bndbox')[1].text = str(400+int(members[0].find('bndbox')[1].text))
            members[0].find('shape')[0][1].text = str(400+int(members[0].find('shape')[0][1].text))
            members[0].find('bndbox')[3].text = str(400+int(members[0].find('bndbox')[3].text))
            members[0].find('shape')[0][3].text = str(400+int(members[0].find('shape')[0][3].text))

            tree.write(xml[:-4] + '_t' + '.xml', 'UTF-8')  # another address
            return 0
        else:
            # mostly a right box
            members[0].find('bndbox')[0].text = str(640)
            members[0].find('shape')[0][0].text = str(640)

            members[0].find('bndbox')[0].text = str(int(members[0].find('bndbox')[0].text) - 640)
            members[0].find('shape')[0][0].text = str(int(members[0].find('shape')[0][0].text) - 640)
            members[0].find('bndbox')[2].text = str(int(members[0].find('bndbox')[2].text) - 640)
            members[0].find('shape')[0][2].text = str(int(members[0].find('shape')[0][2].text) - 640)

            tree.write(xml[:-4] + '_t' + '.xml', 'UTF-8')  # another address
            return 0
    elif len(members) == 2:
        # two boxes, must be left and right members[0] left, members[1] right
        if int(members[0].find('bndbox')[2].text) > 640:
            members[0].find('bndbox')[2].text = str(640)
            members[0].find('points')[2].text = str(640)   # two x
            if int(members[0].find('bndbox')[0].text) > 640:
                members[0].find('bndbox')[0].text = str(640)
                members[0].find('points')[0].text = str(640)
                members[0].find('area')[0].text = str(0)
            else:
                members[0].find('area')[0].text = str((int(members[0].find('bndbox')[2].text) - int(members[0].find('bndbox')[0].text))*(int(members[0].find('bndbox')[3].text) - int(members[0].find('bndbox')[0].text)))
        members[0].find('bndbox')[1].text = str(400 + int(members[0].find('bndbox')[1].text))
        members[0].find('shape')[0][1].text = str(400 + int(members[0].find('shape')[0][1].text))
        members[0].find('bndbox')[3].text = str(400 + int(members[0].find('bndbox')[3].text))
        members[0].find('shape')[0][3].text = str(400 + int(members[0].find('shape')[0][3].text))

        if int(members[1].find('bndbox')[0].text) <= 640:
            members[1].find('bndbox')[0].text = str(640)
            members[1].find('x')[1].text = str(640)   # two x
            if int(members[1].find('bndbox')[2].text) <= 640:
                members[1].find('bndbox')[2].text = str(640)
                members[1].find('shape')[0][0].text = str(640)
                members[1].find('area')[0].text = str(0)
            else:
                members[1].find('area')[0].text = str((int(members[1].find('bndbox')[2].text) - int(
                    members[1].find('bndbox')[0].text)) * (int(members[1].find('bndbox')[3].text) - int(
                    members[1].find('bndbox')[0].text)))
        members[1].find('bndbox')[0].text = str(int(members[1].find('bndbox')[0].text) - 640)
        members[1].find('shape')[0][0].text = str(int(members[1].find('shape')[0][0].text) -640)
        members[1].find('bndbox')[2].text = str(int(members[1].find('bndbox')[2].text) - 640)
        members[1].find('shape')[0][2].text = str(int(members[1].find('shape')[0][2].text) - 640)

    tree.write(xml[:-4] + '_t' + '.xml', 'UTF-8')
    return 0


def main():
    source = r'D:\iqiyi\short_expose\2020-09-27_regress_bad_labeled\labeled'
    store = r''

    for jpg in glob.glob(source + '/*.jpg'):
        flag = tran_xml(jpg)
        tran_pix(jpg, flag)


    # pix, pix_tran = tran_pix(source)
    # tran_xml(source)
    # plt.figure(figsize=(16, 15))
    # plt.subplot(1, 2, 1)
    # plt.imshow(pix)
    # plt.subplot(1, 2, 2)
    # plt.imshow(pix_tran)
    # plt.show()

main()




