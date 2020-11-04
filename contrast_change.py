# change the contrast of pix
# we assume that there will not be a picture and folder in the same folder
# 打开一个文件夹，把所有的文件夹名都存成新地址，所有的图像文件都处理并备份
# 我们现在先默认文件夹层数不知道但是只有底层有文件
import os
import shutil
import cv2
import numpy as np


def contrast_linear_fix(path_source, path_store, alpha=2.0, beta=20):
    image = cv2.imread(path_source)
    if image is None:
        print("wrong source path")
        exit(0)
    image_copy = np.uint8(np.clip((alpha*image + beta), 0, 255))
    # image_copy = np.zeros(image.shape, image.dtype)
    #
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             image_copy[y, x, c] = np.clip(alpha*image[y, x, c] + beta, 0, 255)
    cv2.imwrite(path_store, image_copy)


def contrast_gamma_fix(path_source, path_store, gamma=0.4):
    image = cv2.imread(path_source)
    if image is None:
        print("wrong source path")
        exit(0)
        table = np.empty((1,256), np.uint8)
    for i in range(256):
        table[0, i] = np.clip(pow(i / 256.0, gamma) * 255.0, 0, 255)
    image_copy = cv2.LUT(image, table)
    cv2.imwrite(path_store, image_copy)



def main():
    path_list = []
    path_list_copy = []
    sourece_path = r'D:\iqiyi\DATA\organized2020-11'
    store_path = r'D:\iqiyi\DATA\organized2020-11-contrast'
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
                    jpg_name = i[:-4] + r'.jpg'
                    contrast_linear_fix(temp + "\\" + jpg_name, temp_copy + "\\" + jpg_name)
                    shutil.copy(temp + "\\" + i, temp_copy + "\\" + i)
                    print("one done")
main()

