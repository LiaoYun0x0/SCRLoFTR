
from turtle import Turtle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import math
from .ses_conv  import SESConv_Z2_H, SESConv_H_H,SESConv_H_H_1x1, SESMaxProjection



#def conv1x1(in_channels, out_channel, stride=1, scales=[0.9 * 1.41**i for i in range(3)]):
    #return SESConv_H_H_1x1(in_channels, out_channel,stride=stride, bias=False, num_scales=len(scales))


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, scales=[1.0], pool=False, interscale=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        if pool:
            self.conv1 = nn.Sequential(
                SESMaxProjection(),
                SESConv_Z2_H(in_planes, out_planes, kernel_size=7, effective_size=3,
                             stride=stride, padding=3, bias=False, scales=scales, basis_type='A')
            )
        else:
            
                self.conv1 = SESConv_H_H(in_planes, out_planes, 1, kernel_size=7, effective_size=3, stride=stride,
                                         padding=3, bias=False, scales=scales, basis_type='A')
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = SESConv_H_H(out_planes, out_planes, 1, kernel_size=7, effective_size=3, stride=1,
                                 padding=3, bias=False, scales=scales, basis_type='A')
        #self.conv1x1 =  SESConv_H_H_1x1(in_planes, out_planes,stride=stride, bias=False, num_scales=len(scales))#jiade
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and SESConv_H_H_1x1(in_planes, out_planes,
                                                                      stride=stride, bias=False, num_scales=len(scales)) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, scales=[0.0], pool=False, interscale=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
                                      nb_layers, stride, dropRate, scales, pool, interscale)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, scales, pool, interscale):
        layers = []
        for i in range(nb_layers):#一个block两个basicblock
            pool_layer = pool and (i == 0)
            interscale_layer = interscale and (i == 0)
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, dropRate, scales,
                                pool=pool_layer, interscale=interscale_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=8, dropRate=0.3, scales=[0.9 * 1.41**i for i in range(3)],kernel_size=11, basis_type='A',  
                 pools=[False, True, True], interscale=[False, False, False], **kwargs):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 196, 32 * widen_factor]#通道16，128，196，256
        #assert((depth - 4) % 6 == 0)
        #n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = SESConv_Z2_H(1, nChannels[0], kernel_size=7, effective_size=3, stride=2,
                                  padding=3, bias=False, scales=scales, basis_type='A')
        # 1st block
        self.block1 = NetworkBlock(2, nChannels[0], nChannels[1], block, 1,
                                   dropRate, scales=scales, pool=pools[0], interscale=interscale[0])#1/2 128
        # 2nd block
        self.block2 = NetworkBlock(2, nChannels[1], nChannels[2], block, 2,
                                   dropRate, scales=scales, pool=pools[1], interscale=interscale[1])#1/4 196
        # 3rd block
        self.block3 = NetworkBlock(2, nChannels[2], nChannels[3], block, 2,
                                   dropRate, scales=scales, pool=pools[2], interscale=interscale[2])#1/8  256
        # global average pooling and classifier
        self.proj = SESMaxProjection()
        #self.bn1 = nn.BatchNorm2d(nChannels[3])
        #self.relu = nn.ReLU(inplace=True)
        #self.fc = nn.Linear(nChannels[3], num_classes)
        #self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1)):
                nelement = m.weight.nelement()
                n = nelement / m.in_channels
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
        #FPN
        self.layer3_outconv = SESConv_H_H_1x1(nChannels[3],nChannels[3],stride=1, bias=False, num_scales=len(scales))#256--256
        self.layer2_outconv = SESConv_H_H_1x1(nChannels[2],nChannels[3],stride=1, bias=False, num_scales=len(scales))#196--256
        self.layer2_outconv2 = nn.Sequential(
             SESConv_Z2_H(nChannels[3], nChannels[3],  kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            
            nn.BatchNorm3d(nChannels[3]),

            SESConv_H_H(nChannels[3], nChannels[2], 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            
            nn.ReLU(True),
            
            nn.BatchNorm2d(nChannels[2]),
        )
        self.layer1_outconv = SESConv_H_H_1x1(nChannels[1],nChannels[2],stride=1, bias=False, num_scales=len(scales))
        self.layer1_outconv2 = nn.Sequential(
             SESConv_Z2_H(nChannels[2], nChannels[2], kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            
            nn.BatchNorm3d(nChannels[2]),

            SESConv_H_H(nChannels[2], nChannels[1], 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            
            nn.BatchNorm2d(nChannels[1]),
            
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x1=self.block1(x0)#128 2
        x2=self.block2(x1)#196 4
        x3=self.block3(x2)#256 8
        
        x3_out = self.layer3_outconv(x3)
        x3_out_p = self.proj(x3_out)
        x3_out_2x = F.interpolate(x3_out_p, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out_p = self.proj(x2_out)
        x2_out = self.layer2_outconv2(x2_out_p+x3_out_2x)#这次outconv2后维度不变
        
        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out_p = self.proj(x1_out)
        x1_out = self.layer1_outconv2(x1_out_p+x2_out_2x)
        
        
        
        
        #out = self.block1(out)这里输出四维
        #out = self.block2(out)这里输出四维
        #out = self.block3(out)这里输出四维
        #out = self.proj(out)
        #out = self.relu(self.bn1(out))

        #out = F.adaptive_avg_pool2d(out, 1)
        #out = out.view(-1, self.nChannels)
        #out = self.fc(out)
        return  [x3_out_p, x1_out]




if __name__ == "__main__":
        pass