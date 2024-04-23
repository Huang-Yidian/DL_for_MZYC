from typing import *
import torch
import torch.nn.functional as F

from torch import nn
    
class BasicBlock(nn.Module):
    expansion = 1   # 残差结构中主分支所采用的卷积核的个数是否发生变化。对于浅层网络，每个残差结构的第一层和第二层卷积核个数一样，故是1
    # 定义初始函数
    # in_channel输入特征矩阵深度，out_channel输出特征矩阵深度（即主分支卷积核个数）
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm="group"):   # downsample对应虚线残差结构捷径中的1×1卷积
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1,bias=False)  # 使用bn层时不使用bias
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1,bias=False)  # 实/虚线残差结构主分支中第二层stride都为1
        self.downsample = downsample   # 默认是None
        if norm == "group":
            self.bn1 = nn.GroupNorm(32,out_channel)
            self.bn2 = nn.GroupNorm(32,out_channel) 
        elif norm == "instance":
            self.bn1 = nn.InstanceNorm2d(out_channel)
            self.bn2 = nn.InstanceNorm2d(out_channel) 
        elif norm == "batch":
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel) 

# 定义正向传播过程
    def forward(self, x):
        identity = x   # 捷径分支的输出值
        if self.downsample is not None:   # 对应虚线残差结构
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)   # 这里不经过relu激活函数
        out = out + identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """
    __init__
        in_channel：残差块输入通道数
        out_channel：残差块输出通道数
        stride：卷积步长
        downsample：在_make_layer函数中赋值，用于控制shortcut图片下采样 H/2 W/2
    """
    expansion = 4   # 残差块第3个卷积层的通道膨胀倍率
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm="group"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,bias=False)  
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1,bias=False) 
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1,bias=False) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if norm == "group":
            self.bn1 = nn.GroupNorm(32,out_channel)
            self.bn2 = nn.GroupNorm(32,out_channel)
            self.bn3 = nn.GroupNorm(32,out_channel*self.expansion)
        elif norm == "instance":
            self.bn1 = nn.InstanceNorm2d(out_channel)
            self.bn2 = nn.InstanceNorm2d(out_channel) 
            self.bn3 = nn.InstanceNorm2d(out_channel*self.expansion)
        elif norm == "batch":
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel) 
            self.bn3 = nn.BatchNorm2d(out_channel*self.expansion) 

    def forward(self, x):
        identity = x    # 将原始输入暂存为shortcut的输出
        if self.downsample is not None:
            identity = self.downsample(x)   # 如果需要下采样，那么shortcut后:H/2，W/2。C: out_channel -> 4*out_channel(见ResNet中的downsample实现)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity    
        out = self.relu(out)
        return out
    
    

class ResNet(nn.Module):
    def __init__(self,
                 inch,
                 block,   # 残差结构，Basicblock or Bottleneck
                 ch,
                 blocks_num,
                 strides,
                 norm = "group"):   # 为了能在ResNet网络基础上搭建更加复杂的网络，默认为True
        super(ResNet, self).__init__()
        self.in_channel = 64   # 通过max pooling之后所得到的特征矩阵的深度
        self.conv1 = nn.Conv2d(inch, self.in_channel, kernel_size=7, stride=2,
                               padding=3,bias=False)   # 输入特征矩阵的深度为3（RGB图像），高和宽缩减为原来的一半
        self.norm = norm
        if norm == "group":
            self.bn1 = nn.GroupNorm(32,self.in_channel)
        elif norm == "instance":
            self.bn1 = nn.InstanceNorm2d(self.in_channel)
        elif norm == "batch":
            self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True) #
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 高和宽缩减为原来的一半
        self.layer1 = self._make_layer(block, ch[0],  blocks_num[0], stride=strides[0])   # 对应conv2_x
        self.layer2 = self._make_layer(block, ch[1], blocks_num[1], stride=strides[1])   # 对应conv3_x
        self.layer3 = self._make_layer(block, ch[2], blocks_num[2], stride=strides[2])   # 对应conv4_x
        self.layer4 = self._make_layer(block, ch[3], blocks_num[3], stride=strides[3])   # 对应conv5_x


    def _make_layer(self, block, channel, block_num, stride=1):   # stride默认为1
        # block即BasicBlock/Bottleneck
        # channel即残差结构中第一层卷积层所使用的卷积核的个数
        # block_num即该层一共包含了多少层残差结构
        downsample = None

        # 左：输出的高和宽相较于输入会缩小；右：输入channel数与输出channel数不相等
        # 两者都会使x和identity无法相加
        if stride != 1 or self.in_channel != channel * block.expansion:  # ResNet-18/34会直接跳过该if语句（对于layer1来说）
            # 对于ResNet-50/101/152：
            # conv2_x第一层也是虚线残差结构，但只调整特征矩阵深度，高宽不需调整
            # conv3/4/5_x第一层需要调整特征矩阵深度，且把高和宽缩减为原来的一半
            if self.norm == "group":
                norm_layer = nn.GroupNorm(32,channel * block.expansion)
            elif self.norm == "instance":
                norm_layer = nn.InstanceNorm2d(channel * block.expansion)
            elif self.norm == "batch":
                norm_layer = nn.BatchNorm2d(channel * block.expansion)
                
            downsample = nn.Sequential(       # 下采样
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride,bias=False),
                norm_layer 
                )
        layers = []
        layers.append(block(self.in_channel,  # 输入特征矩阵深度，64
                            channel,  # 残差结构所对应主分支上的第一个卷积层的卷积核个数
                            downsample=downsample,
                            stride=stride,
                            norm=self.norm))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):   # 从第二层开始都是实线残差结构
            layers.append(block(self.in_channel,  # 对于浅层一直是64，对于深层已经是64*4=256了
                                channel,norm=self.norm))  # 残差结构主分支上的第一层卷积的卷积核个数
        # 通过非关键字参数的形式传入nn.Sequential
        return nn.Sequential(*layers)   # *加list或tuple，可以将其转换成非关键字参数，将刚刚所定义的一切层结构组合在一起并返回

# 正向传播过程
    def forward(self, x):
        x = self.conv1(x)   # 7×7卷积层
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 3×3 max pool

        x = C2 = self.layer1(x) 
        x = C3 = self.layer2(x) 
        x = C4 = self.layer3(x) 
        x = C5 = self.layer4(x) 
        return x , [C2,C3,C4,C5]
    
    
def ResNet18(inch):
    return ResNet(inch, BasicBlock, [64,128,256,512],[2,2,2,2], [1,2,2,2])

def ResNet34(inch):
    return ResNet(inch, BasicBlock, [64,128,256,512],[3,4,6,3], [1,2,2,2])

def ResNet50(inch):
    return ResNet(inch, Bottleneck, [64,128,256,512],[3,4,6,3], [1,2,2,2])

def ResNet101(inch):
    return ResNet(inch, Bottleneck, [64,128,256,512],[3,4,23,3], [1,2,2,2])
