import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil


data_img_path = './data_source'
data_mask_path = './data_seg'
data_target = './data'

classes = ['nCT', 'pCT', 'NiCT']

data_img_class_list = [data_img_path + os.sep + c for c in classes]

data_img_list = []

for i in range(len(classes)):
    data_img_list.append(os.listdir(data_img_class_list[i]))

data_n_img_path = [data_img_path + os.sep + classes[0] + os.sep + i for i in data_img_list[0]]
data_p_img_path = [data_img_path + os.sep + classes[1] + os.sep + i for i in data_img_list[1]]
data_Ni_img_path = [data_img_path + os.sep + classes[2] + os.sep + i for i in data_img_list[2]]

random.shuffle(data_n_img_path)
random.shuffle(data_p_img_path)
random.shuffle(data_Ni_img_path)

train_num_n = int(len(data_n_img_path) / 10 * 8)
val_num_n = int(len(data_n_img_path) / 10)
test_num_n = int(len(data_n_img_path) / 10)

data_img_train = data_n_img_path[:int(len(data_n_img_path) / 10 * 8)]\
                + data_p_img_path[:int(len(data_p_img_path) / 10 * 8)] \
                + data_Ni_img_path[:int(len(data_Ni_img_path) / 10 * 8)]

data_img_val = data_n_img_path[int(len(data_n_img_path) / 10 * 8):int(len(data_n_img_path) / 10 * 9)]\
                + data_p_img_path[int(len(data_p_img_path) / 10 * 8):int(len(data_p_img_path) / 10 * 9)]\
                + data_Ni_img_path[int(len(data_Ni_img_path) / 10 * 8):int(len(data_Ni_img_path) / 10 * 9)]

data_img_test = data_n_img_path[int(len(data_n_img_path) / 10 * 9):]\
                + data_p_img_path[int(len(data_p_img_path) / 10 * 9):]\
                + data_Ni_img_path[int(len(data_Ni_img_path) / 10 * 9):]

data_mask_train = [s.replace(data_img_path, data_mask_path) for s in data_img_train]
data_mask_val = [s.replace(data_img_path, data_mask_path) for s in data_img_val]
data_mask_test = [s.replace(data_img_path, data_mask_path) for s in data_img_test]
# img = Image.open(data_mask_train[0]).convert('L')
# plt.imshow(img,cmap='gray')

# img train
for i in range(len(data_img_train)):
    try:
        shutil.copy(data_img_train[i], data_target + '/train/' + data_img_train[i].split('/')[-1])
    except:
        print("Unexpected error:", sys.exc_info())

# img val
for i in range(len(data_img_val)):
    try:
        shutil.copy(data_img_val[i], data_target + '/val/' + data_img_val[i].split('/')[-1])
    except:
        print("Unexpected error:", sys.exc_info())

# img test
for i in range(len(data_img_test)):
    try:
        shutil.copy(data_img_test[i], data_target + '/test/' + data_img_test[i].split('/')[-1])
    except:
        print("Unexpected error:", sys.exc_info())

# mask train
for i in range(len(data_mask_train)):
    try:
        shutil.copy(data_mask_train[i], data_target + '/train_mask/' + data_mask_train[i].split('/')[-1])
    except:
        print("Unexpected error:", sys.exc_info())

# mask val
for i in range(len(data_mask_val)):
    try:
        shutil.copy(data_mask_val[i], data_target + '/val_mask/' + data_mask_val[i].split('/')[-1])
    except:
        print("Unexpected error:", sys.exc_info())

# mask test
for i in range(len(data_mask_test)):
    try:
        shutil.copy(data_mask_test[i], data_target + '/test_mask/' + data_mask_test[i].split('/')[-1])
    except:
        print("Unexpected error:", sys.exc_info())

class data_split():
    def __init__(self) -> None:
        print('import module success!')