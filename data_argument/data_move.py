import os
import sys
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data_path = './data_seg/nCT1.jpg'
data_path_1 = '/mnt/data/shanliang/program/HUST-19_2020_08_18/nCT/nCT1.jpg'

img = Image.open(data_path).convert('L')
img_1 = Image.open(data_path_1).convert('RGB')

img1 = np.array(img)
img_2 = np.array(img_1)
img2 = img1[:,:,1]


plt.imshow(img_1)
plt.show

image_p_path = '/mnt/data/shanliang/program/HUST-19_2020_08_18/pCT/'
image_n_path = '/mnt/data/shanliang/program/HUST-19_2020_08_18/nCT/'
mask_path = './data_seg/'

num = 5
mask_list = os.listdir(mask_path)
mask_n_list = [n for n in mask_list if n[0] == 'n' and n[-4:] == '.jpg']
mask_p_list = [n for n in mask_list if n[0] == 'p' and n[-4:] == '.jpg']
mask_n_used_list = random.sample(mask_n_list, num)
mask_p_used_list = random.sample(mask_p_list, num)

mask_n_path_list = [mask_path + i for i in mask_n_used_list]
mask_p_path_list = [mask_path + i for i in mask_p_used_list]
image_n_path_list = [image_n_path + i for i in mask_n_used_list]
image_p_path_list = [image_p_path + i for i in mask_p_used_list]

# adding exception handling
for i in range(50):
    try:
        shutil.copy(image_n_path_list[i], './data/val/' + mask_n_used_list[i])
        shutil.copy(mask_n_path_list[i], './data/val_mask/' + mask_n_used_list[i])
        shutil.copy(image_p_path_list[i], './data/val/' + mask_p_used_list[i])
        shutil.copy(mask_p_path_list[i], './data/val_mask/' + mask_p_used_list[i])
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())

for i in range(len(mask_n_list)):
    try:
        shutil.copy(mask_path + mask_n_list[i], './data_seg/nCT/' + mask_n_list[i])
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())

test_n = os.listdir('./data_seg/nCT/')
test_p = os.listdir('./data_seg/pCT/')
test_Ni = os.listdir('./data_seg/')
test_Ni = [i for i in test_Ni if i[-4:] == '.jpg']

for i in range(len(test_Ni)):
    try:
        os.remove(mask_path + test_Ni[i])
    except OSError as e:
        print("Unable to remove file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())
