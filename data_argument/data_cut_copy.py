import os
import shutil

train_list = os.listdir('../data_cut/train')
train_mask_list = os.listdir('../data_cut/train_mask')
val_list = os.listdir('../data_cut/val')
val_mask_list = os.listdir('../data_cut/val_mask')
test_list = os.listdir('../data_cut/test')
test_mask_list = os.listdir('../data_cut/test_mask')

# yanzheng 
if train_list == train_mask_list and \
    val_list == val_mask_list and \
        test_list == test_mask_list:
        print('data compelete')

# save list
with open('./train_list.txt', 'w+') as f:
    for i in range(len(train_list)):
        f.write(train_list[i] + '\n')

with open('./val_list.txt', 'w+') as f:
    for i in range(len(val_list)):
        f.write(val_list[i] + '\n')       

with open('./test_list.txt', 'w+') as f:
    for i in range(len(test_list)):
        f.write(test_list[i] + '\n')       