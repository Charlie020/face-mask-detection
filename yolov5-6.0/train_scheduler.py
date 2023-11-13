import os

# 第一个命令
command_1 = 'python train.py --resume runs/train/Baseline+DRConv2d/weights/last.pt --cfg models/yolov5s+DRConv2d.yaml'
os.system(command_1)

# 第二个命令
# command_2 = 'python train.py --resume runs/train/Baseline+C2f/weights/last.pt --cfg models/yolov5s+C2f.yaml'
# os.system(command_2)
#
# command_3 = 'python train.py --cfg models/yolov5s+CBAM.yaml'
# os.system(command_3)
