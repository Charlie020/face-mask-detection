import os

# LabelFolderPath = "VOCdevkit/labels/test/"
# LabelFileName = os.listdir(LabelFolderPath)   # 存放label的文件夹中每个label文件的名字组成的字典
# LabelFileLen = len(LabelFileName)       # 长度
#
# nomask = 0
# mask = 0
#
# for filename in LabelFileName:
#     file_path = os.path.join(LabelFolderPath, filename)
#     if os.path.isfile(file_path):
#         with open(file_path, "r") as file:
#             for line in file:
#                 if line[0] == '0':
#                     nomask += 1
#                 elif line[0] == '1':
#                     mask += 1
#
# print(nomask, mask)

LabelIndexFilePath = "E:\\ETemp\\DataSet\\MaskDetection_FromInternet_YOLO\\test.txt"
LabelFolderPath = "E:\\ETemp\\DataSet\\MaskDetection_FromInternet_YOLO\\labels\\"

LabelFileName = os.listdir(LabelFolderPath)
LabelFileLen = len(LabelFileName)

nomask = mask = 0

with open(LabelIndexFilePath, "r") as file:
    for line in file:
        line = line.strip()
        line = line.replace("E:/PythonCode/MaskDetect_YOLOv5-6.0/yolov5-6.0/VOCdevkit/images/", "")
        line = line.rstrip('.jpg')
        line += '.txt'
        # print(line)
        if (line in LabelFileName):
            with open(LabelFolderPath + line, "r") as label:
                for line2 in label:
                    if line2[0] == '0':
                        nomask += 1
                    elif line2[0] == '1':
                        mask += 1

print(nomask, mask)
