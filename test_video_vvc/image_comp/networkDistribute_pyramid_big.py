import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign

import numpy as np

class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.conv = nn.Conv2d(
            1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        #print('in:',input.shape)
        x = self.conv(input)
        #print('x:',x.shape)
        # input [batch, 3, 144, 176]
        # x     [batch, 64, 72, 88]
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        # hidden1 tuple:2
        # x       [12, 256, 36, 44]
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        # hidden2 tuple:2
        # x       [12, 512, 18, 22]

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        # hidden2 tuple:2
        # x       [12, 512, 9, 11]

        return x, hidden1, hidden2, hidden3

class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = torch.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    # 1210 添加参数bits
    # 1229 去掉参数bits 对应更新BPP的计算
    def __init__(self):
        super(DecoderCell, self).__init__()

        self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.rnn1 = ConvLSTMCell(
            (512 + 1) , # 边信息加在第一层
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            (128 + 1) ,  # 边信息加在第二层
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            (128 + 1) ,  # 边信息加在第三层
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = ConvLSTMCell(
            (64 + 1),  # 边信息加在第四层
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.conv2 = nn.Conv2d(
            (32 + 1) ,  # 边信息加在第四层
            1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4, dataSide):
        x = self.conv1(input)  # 32, 512, 2, 2
        
        y0 = F.interpolate(dataSide, size=[dataSide.size(2) // 16, dataSide.size(3) // 16])
        #print('shape:',x.shape, y0.shape)
        x = torch.cat([x, y0], dim=1)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        # 0318 添加边信息在不同的解码层中
        
        y0 = F.interpolate(dataSide, size=[dataSide.size(2) // 8, dataSide.size(3) // 8])
        #print('shape:',x.shape, y0.shape)
        x = torch.cat([x, y0], dim=1)  
                
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        
        y0 = F.interpolate(dataSide, size=[dataSide.size(2) // 4, dataSide.size(3) // 4])
        x = torch.cat([x, y0], dim=1)      
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

       
        y0 = F.interpolate(dataSide, size=[dataSide.size(2) // 2, dataSide.size(3) // 2])
        x = torch.cat([x, y0], dim=1)
        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]  # 32, 128, 16,16
        x = F.pixel_shuffle(x, 2)  # 32, 32, 32, 32

       
        x = torch.cat([x, dataSide], dim=1)
        x = torch.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4
