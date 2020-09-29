import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        for member in tree.findall('object'):
            value = (tree.find('filename').text,
                     int(tree.find('size')[0].text),
                     int(tree.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(float(member[4][1].text)),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    csv_root = r'D:\iqiyi\short_expose\labeled_20_9_25\csv_20_9_25'
    annotation_root = r'D:\iqiyi\short_expose\labeled_20_9_25\prepared_20_9_25'
    for directory in ['train', 'test', 'validation']:
        xml_path = os.path.join(annotation_root, directory)
        xml_df = xml_to_csv(xml_path)
        xml_df.to_csv(csv_root + '/ball_{}_lables.csv'.format(directory), index=None)
        print('success')
main()