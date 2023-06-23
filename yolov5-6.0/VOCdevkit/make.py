# 划分VOC数据集，并保存至ImageSets下Main中的各TXT文件

import os
import random

trainval_percent = 0.8             # train和val一共占数据集的比例  这里为0.8
train_percent = 0.75               # train占train+val的比例      这里为0.75，相当于总数据集的0.8*0.75=0.6
xmlfilepath = 'Annotations/'
txtsavepath = 'ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num*trainval_percent)
tr = int(tv*train_percent)
trainval = random.sample(list,tv)
train = random.sample(trainval,tr)

ftrainval = open(txtsavepath+'/trainval.txt', 'w')
ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
print('Well Done！！！')

