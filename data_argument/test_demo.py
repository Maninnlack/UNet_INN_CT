import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm

# img_dir = '/mnt/data/shanliang/program/HUST-19_2020_08_18/nCT'
img_seg_dir = '/mnt/data/shanliang/program/HUST-19_2020_08_18/NiCT_i'
output_dir = '/mnt/data/shanliang/program/UNet_INN/data_seg/NiCT/'
# img_path_list = [img_dir + os.sep + path for path in os.listdir(img_dir)]
img_seg_path_list = [img_seg_dir + os.sep + path for path in os.listdir(img_seg_dir)]
# img_path_list = img_path_list[:5]

for img_seg_path in tqdm(img_seg_path_list):

    img_name = img_seg_path.split('/')[-1]
    #img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    img_seg = cv2.imdecode(np.fromfile(img_seg_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    img_seg_trans = np.ones_like(img_seg)
    img_seg_trans[img_seg < 10] = 0
    img_seg_trans = img_seg_trans * 255
    
    # plt.imsave(output_dir + img_name, img_seg_trans, cmap='gray')
    cv2.imencode('.jpg', img_seg_trans)[1].tofile(output_dir + img_name)