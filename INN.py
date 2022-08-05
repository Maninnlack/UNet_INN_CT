import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ImageDataset
from Unet_INN import UNet_INN
from utils.loss import FocalLoss

# image and mask path
train_img_path = './data/train'
train_mask_path = './data/train_mask'
val_img_path = './data/val'
val_mask_path = './data/val_mask'

# data_test image and mask path
# train_img_path = './data_test/train'
# train_mask_path = './data_test/label'
# val_img_path = './data_test/val'
# val_mask_path = './data_test/val_label'

checkpoint_path = './model_save'
output_path = './output/INN'
log_path = './log'

#  nvidia configure
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# hyper perameters
batch_size = 4
epochs = 5
learning_rate = 1e-3

# dataset
train_dataset = ImageDataset(train_img_path, train_mask_path, dtype='train')
val_dataset = ImageDataset(val_img_path, val_mask_path, dtype='val')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# net init
model_name = 'INN_UNet_data'
net = UNet_INN(in_channels=3, out_channels=1)
net.to(device=device)

# optimizer, loss function, learning rate
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = FocalLoss()
# 损失函数可选 Diceloss， Focalloss 
# 当最后一层没有加sigmoid时，loss函数要选F.binary_cross_entropy_with_logits

batch_num = 0
train_losses, val_losses = [], [] # 记录全程loss 变化
log = []  # 写入txt记录

# tensorboard 可视化
writer = SummaryWriter()
writer.add_graph(net, torch.rand([batch_size,3,256,256]).to(device))

# train
for epoch in range(epochs):
    net.train() 
    for index, batch in enumerate(train_dataloader):
        image = batch['img'].to(device=device)
        mask = batch['mask'].to(device=device)

        assert image.shape[1] == net.in_channels,\
            f'input channels:{net.in_channels} is not equal to image channels:{image.shape[1]}'

        pred_mask = net(image)
        loss = criterion(pred_mask, mask)    # target mask需要long格式
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        train_losses.append(train_loss) # item防止爆显存
        current = index * len(image) + 1
        size_train = len(train_dataloader.dataset)
        writer.add_scalar('train loss', train_loss, batch_num)

        # validation
        size_val = len(val_dataloader.dataset)
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                image = batch['img'].to(device)
                mask = batch['mask'].to(device)
                pred_mask = net(image)
                val_loss += criterion(pred_mask, mask).item()
        val_loss = val_loss / size_val
        val_losses.append(val_loss)
        writer.add_scalar('val loss', val_loss, batch_num)
        batch_num += 1
        
        print(f'train loss: {train_loss:>7f} val loss: {val_loss:>7f} [{current:>5d}/{size_train:>5d}]  epoch:{epoch}/{epochs}')
        log.append(f'train loss: {train_loss:>7f} val loss: {val_loss:>7f} [{current:>5d}/{size_train:>5d}]  epoch:{epoch}/{epochs}\n')

with open('./INN_test.txt', 'a+') as f:
    f.write(model_name + '\n')
    for i in range(len(log)):
        f.write(log[i])


torch.save(net.state_dict(), checkpoint_path + os.sep + model_name + '.pth')
print(f'model {model_name} saved!')

# writer.flush()
writer.close()
