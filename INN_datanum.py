import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ImageDataset
from Unet_INN import UNet_INN
from spikingjelly.clock_driven import functional
from utils.loss import FocalLoss 


def main():
    # image and mask path
    train_img_path = './data/train'
    train_mask_path = './data/train_mask'
    val_img_path = './data/val'
    val_mask_path = './data/val_mask'

    checkpoint_path = './model_save'
    # output_path = './output/INN'
    # log_path = './log'

    parse = argparse.ArgumentParser(description='UNet-INN')
    parse.add_argument('--device', default='cuda:0', help='运行的设备')
    parse.add_argument('-b', '--batchsize', default=8, type=int, help='Batch 大小')
    parse.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parse.add_argument('-e', '--epoch', default=10, type=int, help='epoch数量')
    parse.add_argument('-T', default=5, type=int)

    arg = parse.parse_args()

    #  nvidia configure
    device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # hyper perameters
    batch_size = arg.batchsize
    epochs = arg.epoch
    learning_rate = arg.learning_rate

    # net init
    net = UNet_INN(in_channels=3, out_channels=1)
    net.to(device=device)

    # optimizer, loss function, learning rate
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = FocalLoss(logits=True)
    # 损失函数可选 Diceloss， Focalloss 
    # 当最后一层没有加sigmoid时，loss函数要选F.binary_cross_entropy_with_logits

    # dataset for different data number
    for data_num in range(10,101, 10):
        if data_num == 0:
            train_dataset = ImageDataset(train_img_path, train_mask_path, dtype='train')
            val_dataset = ImageDataset(val_img_path, val_mask_path, dtype='val')

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            train_dataset = ImageDataset(train_img_path, train_mask_path, dtype='train', dnum=data_num)
            val_dataset = ImageDataset(val_img_path, val_mask_path, dtype='val', dnum=data_num)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        model_name = f'INN_UNet_data_{data_num}'
        print(model_name)
        
        try:
            train(epochs, train_dataloader, val_dataloader, device, 
                net, criterion, optimizer, model_name, checkpoint_path)
        # saving model when interrupting training
        except KeyboardInterrupt:
            train_time = time.strftime("%y-%m-%d", time.localtime())
            torch.save(net.state_dict(), checkpoint_path + os.sep + model_name + '_' + train_time + '.pth')
            print(f'Interruptted with model {model_name} saved.')


def train(epochs, 
        train_dataloader, 
        val_dataloader, 
        device, 
        net, 
        criterion,
        optimizer,
        model_name,
        checkpoint_path):

        train_losses, val_losses = [], [] # 记录全程loss 变化
        log = []  # 写入txt记录

        data_num = model_name.split('_')[-1]
        # tensorboard 可视化
        writer = SummaryWriter(comment=data_num)
        # writer.add_graph(net, (torch.rand([batch_size,3,256,256]).to(device)))

        # train
        for epoch in range(epochs):
            net.train() 
            epoch_train_loss = 0
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
                # 脉冲神经网络需要清空状态重置
                functional.reset_net(net)
                
                train_loss = loss.item()
                train_losses.append(train_loss) # item防止爆显存
                epoch_train_loss += train_loss
                current = index * len(image) + 1
                size_train = len(train_dataloader.dataset)
                writer.add_scalar('train loss', train_loss, current + epoch * len(train_dataloader))
                print(f'train loss: {train_loss:>7f} [{current:>5d}/{size_train:>5d}]  epoch:{epoch}/{epochs}')
                log.append(f'train loss: {train_loss:>7f} [{current:>5d}/{size_train:>5d}]  epoch:{epoch}/{epochs}\n')

            writer.add_scalar('epoch average train loss', epoch_train_loss / len(train_dataloader), epoch)
            print(f'epoch average train loss: {epoch_train_loss / len(train_dataloader)} epoch:{epoch}/{epochs}')
            log.append(f'epoch average train loss: {epoch_train_loss / len(train_dataloader)} epoch:{epoch}/{epochs}\n')

            # validation
            size_val = len(val_dataloader)
            net.eval()
            val_loss = 0
            with torch.no_grad():
                for _, batch in enumerate(val_dataloader):
                    image = batch['img'].to(device)
                    mask = batch['mask'].to(device)
                    pred_mask = net(image)
                    val_loss += criterion(pred_mask, mask).item()
                    # 脉冲神经网络需要清空状态重置
                    functional.reset_net(net)
            val_loss = val_loss / size_val
            val_losses.append(val_loss)
            writer.add_scalar('val loss', val_loss, epoch)
            
            print(f'val loss: {val_loss:>7f}  epoch:{epoch}/{epochs}')
            log.append(f'val loss: {val_loss:>7f}  epoch:{epoch}/{epochs}\n')

        print(f'minimum train loss: {min(train_losses):>7f} minimum val loss: {min(val_losses):>7f}  epoch:{epoch}/{epochs}')
        log.append(f'minimum train loss: {min(train_losses):>7f} minimum val loss: {min(val_losses):>7f}  epoch:{epoch}/{epochs}\n')

        with open('./INN_test.txt', 'a+') as f:
            f.write(model_name + '\n')
            for i in range(len(log)):
                f.write(log[i])

        torch.save(net.state_dict(), checkpoint_path + os.sep + model_name + '.pth')
        print(f'model {model_name} saved!')

        writer.close()


if __name__ == '__main__':
    main()