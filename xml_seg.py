import os
import random
import time
import shutil
# use the path of all labeled xml
xml_source_path = r'D:\iqiyi\short_expose\labeled_20_9_25\prepared_20_9_25\annotations'
xml_segment_path = r'D:\iqiyi\short_expose\labeled_20_9_25\prepared_20_9_25'

trainval_percent = 0.9
train_percent = 0.85
random.seed(0)  # make sure every iteration of this demo using the same data segmentation

total_xml = os.listdir(xml_source_path)
num = len(total_xml)
list = range(len(total_xml))
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
print("val size", tv - tr)
print("train size", tr)

test_num = 0
val_num = 0
train_num = 0

for i in list:
    name = total_xml[i]
    if i in trainval:
        if i in train:
            directory = "train"
            train_num += 1
            xml_path = os.path.join(xml_segment_path, directory)
            print(xml_path)
            if not os.path.exists(xml_path):
                os.mkdir(xml_path)
            filepath = os.path.join(xml_source_path, name)
            newfile = os.path.join(xml_segment_path, os.path.join(directory, name))
            shutil.copyfile(filepath, newfile)
        else:
            directory = "validation"
            val_num += 1
            xml_path = os.path.join(xml_segment_path, directory)
            print(xml_path)
            if not os.path.exists(xml_path):
                os.mkdir(xml_path)
            filepath = os.path.join(xml_source_path, name)
            newfile = os.path.join(xml_segment_path, os.path.join(directory, name))
            shutil.copyfile(filepath, newfile)
    else:
        directory = "test"
        test_num += 1
        xml_path = os.path.join(xml_segment_path, directory)
        print(xml_path)
        if not os.path.exists(xml_path):
            os.mkdir(xml_path)
        filepath = os.path.join(xml_source_path, name)
        newfile = os.path.join(xml_segment_path, os.path.join(directory, name))
        shutil.copyfile(filepath, newfile)

print("trian total", train_num)
print("val total", val_num)
print("test total", test_num)
