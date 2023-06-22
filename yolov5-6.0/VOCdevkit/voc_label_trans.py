import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test', 'val']
# 仅需修改以下路径
Imgpath = 'E:\PythonCode\MaskDetect_YOLOv5-6.0\yolov5-6.0\VOCdevkit\JPEGImages'                   # VOC的JPEGImages文件夹路径
xmlfilepath = 'E:\PythonCode\MaskDetect_YOLOv5-6.0\yolov5-6.0\VOCdevkit\Annotations'              # VOC的Annotations文件夹路径
ImageSets_path = 'E:\PythonCode\MaskDetect_YOLOv5-6.0\yolov5-6.0\VOCdevkit\ImageSets\Main'        # VOC中ImagesSets的Main文件夹的路径
Label_path = 'E:\PythonCode\MaskDetect_YOLOv5-6.0\yolov5-6.0\VOCdevkit'                           # 将XML格式的标签转化为TXT格式后保存的路径
classes = ["nomask", "mask"]                                                                      # 类别名



def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open(xmlfilepath + '\%s.xml' % (image_id), encoding='utf-8')
    out_file = open(Label_path + '\labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        if obj.find('difficult'):
            diff = obj.find('difficult').text
        else:
            diff = 0
        difficult = diff
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


for image_set in sets:
    if not os.path.exists(Label_path + '\labels\\'):
        os.makedirs(Label_path + '\labels\\')
    image_ids = open(ImageSets_path + '\%s.txt' % (image_set)).read().strip().split()
    list_file = open(Label_path + '\%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(Imgpath + '\%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
