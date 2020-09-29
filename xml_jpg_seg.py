# 原始数据xml文件和jpg文件是交叉混合在一起的，我们先将他们从
# D:\iqiyi\short_expose\labeled_20_9_25\labeled_20_9_25
# 分开到 D:\iqiyi\short_expose\labeled_20_9_25\prepared_20_9_25\annotations
# 和 D:\iqiyi\short_expose\labeled_20_9_25\prepared_20_9_25\jpgs

import os
import shutil
source = r'D:\iqiyi\short_expose\labeled_20_9_25\labeled_20_9_25'

ann_path = r'D:\iqiyi\short_expose\labeled_20_9_25\prepared_20_9_25\annotations'
jpg_path = r'D:\iqiyi\short_expose\labeled_20_9_25\prepared_20_9_25\jpgs'

xml_jpg = os.listdir(source)
for item in xml_jpg:
    if item[-3:] == 'jpg':
        filepath = os.path.join(source, item)
        newfile = os.path.join(jpg_path, item)
        shutil.copyfile(filepath, newfile)
    elif item[-3:] == 'xml':
        filepath = os.path.join(source, item)
        newfile = os.path.join(ann_path, item)
        shutil.copyfile(filepath, newfile)
    else:
        print("wrong logic")
