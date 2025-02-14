import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet, BasicBlock, Bottleneck

# for make conv6 layer utilization
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# self refinement module
class SelfRef(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SelfRef,self).__init__()
        self.conv1=conv3x3(in_planes=in_planes,out_planes=out_planes)
        self.bn1=nn.BatchNorm2d(out_planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=conv3x3(in_planes=out_planes,out_planes=out_planes)
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu2=nn.ReLU(inplace=True)

    def forward(self,x):
        identity = x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)
        identity=torch.mul(identity,out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)
        identity=torch.add(identity,out)
        return identity

# channel attention
class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.convlayer = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel,channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(channel),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.atrous_block1 = nn.Conv2d(channel, channel, 1, 1)
        self.atrous_block3 = nn.Conv2d(channel, channel, 3, 1, padding=3, dilation=3)
        self.atrous_block5 = nn.Conv2d(channel, channel, 3, 1, padding=5, dilation=5)
        self.atrous_block7 = nn.Conv2d(channel, channel, 3, 1, padding=7, dilation=7)
 
        self.conv_1x1_output = nn.Conv2d(channel * 4, channel, 1, 1)
    
    def forward(self, x):
        identity=x

        atrous_block1 = self.atrous_block1(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block5 = self.atrous_block5(x)
        atrous_block7 = self.atrous_block7(x)
        x_out = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block3,
                                              atrous_block5, atrous_block7], dim=1))
        y = self.avgpool(x_out)
        y = self.convlayer(y)
        y = identity * y.expand_as(identity)
        return y

# spatial attention
class SpatialAttention(nn.Module):
    def __init__(self, in_channel, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.seq=nn.Sequential(
            nn.Conv2d(1,1,kernel_size=3,padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1,1,kernel_size=3,padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(in_channel * 4, in_channel, 1, 1)

    def forward(self, x):
        identity=x

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        x_out = torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        x_out = self.conv_1x1_output(x_out)

        avgout = torch.mean(x_out, dim=1, keepdim=True)
        maxout, _ = torch.max(x_out, dim=1, keepdim=True)
        avgout = self.seq(avgout)
        maxout = self.seq(maxout)
        x_out = torch.cat([avgout, maxout], dim=1)
        x_out = self.conv(x_out)
        x_out = self.sigmoid(x_out)
        x_out = x_out.expand_as(identity)
        x_out = identity * x_out
        return x_out

# multi-interactive refinement network
class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4=nn.BatchNorm2d(64)
        self.relu4=nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv5=nn.Conv2d(64,64,3,padding=1)
        self.bn5=nn.BatchNorm2d(64)
        self.relu5=nn.ReLU(inplace=True)

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.cs = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self,x):

        hx = x
        hx = self.conv0(hx)  # 8 64 320

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx1 = self.pool1(hx1) # 8 64 160

        hx2 = self.relu2(self.bn2(self.conv2(hx1)))
        hx2 = self.pool2(hx2) # 8 64 80

        hx3 = self.relu3(self.bn3(self.conv3(hx2)))
        hx3 = self.pool3(hx3) # 8 64 40

        hx4 = self.relu4(self.bn4(self.conv4(hx3)))
        hx4=self.pool4(hx4)

        hx5=self.relu5(self.bn5(self.conv5(hx4)))
        hx5=self.pool5(hx5)

        hx5up=F.interpolate(hx5,size=hx4.shape[2:],mode='bilinear')
        h4xfuse=hx4*hx5up
        h4xfuse=self.cs(hx4+h4xfuse)

        hx4up=F.interpolate(hx4,size=hx3.shape[2:],mode='bilinear')
        h3xfuse=hx3*hx4up
        h3xfuse=self.cs(hx3+h3xfuse)
        
        hx3up=F.interpolate(hx3,size=hx2.shape[2:],mode='bilinear')
        h2xfuse=hx2*hx3up
        h2xfuse=self.cs(hx2+h2xfuse)  #8 64 80 80 

        hx2up=F.interpolate(hx2,size=hx1.shape[2:],mode='bilinear')
        h1xfuse=hx1*hx2up
        h1xfuse=self.cs(hx1+h1xfuse)  #8 64 160 160

        h4xfuseup=F.interpolate(h4xfuse,size=h1xfuse.shape[2:],mode='bilinear')
        h3xfuseup=F.interpolate(h3xfuse,size=h1xfuse.shape[2:],mode='bilinear')        
        h2xfuseup=F.interpolate(h2xfuse,size=h1xfuse.shape[2:],mode='bilinear')
        h1xfinal=self.cs(h1xfuse+h2xfuseup+h3xfuseup+h4xfuseup)

        d1=self.upscore2(h1xfinal)
        residual = self.conv_d0(d1)

        return x + residual

# EDMR
class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.config = config
        self.encoder = encoder

        self.inplanes=2048
        self.base_width = 64
        self.conv6 = nn.Sequential(self._make_layer(2048 // 4, 2, stride=2),)
        
        # feat: [64 128 256 512 1024 2048]
        self.post1 = nn.Conv2d(feat[0],64,3,padding=1) # 16
        self.post2 = nn.Conv2d(feat[1],128,3,padding=1) # 16
        self.post3 = nn.Conv2d(feat[2],256,3,padding=1) # 16
        self.post4 = nn.Conv2d(feat[3],512,3,padding=1) # 16 1024
        self.post5 = nn.Conv2d(2048,512,3,padding=1)

        self.maxpool2d = nn.MaxPool2d(2,2,ceil_mode=True)

        # Deep enhancement module
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)
        
        self.atrous_block1 = nn.Conv2d(512, 512, 1, 1)
        self.atrous_block3 = nn.Conv2d(512, 512, 3, 1, padding=6, dilation=6)
        self.atrous_block5 = nn.Conv2d(512, 512, 3, 1, padding=12, dilation=12)
        self.atrous_block7 = nn.Conv2d(512, 512, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(512 * 5, 512, 1, 1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        ## -------------Decoder--------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)
        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)
        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)
        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)
        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        #stage 4d
        self.conv4d_1 = nn.Conv2d(1024,512,3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)
        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)
        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(512,256,3,padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)
        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)
        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        #stage 2d
        self.conv2d_1 = nn.Conv2d(256,128,3,padding=1) # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)
        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)
        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)
        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Double Bilinear Upsampling--------------
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)

        ## -------------Refine Module-------------
        self.selfref6 = SelfRef(512,512)
        self.selfref5 = SelfRef(512,512)
        self.selfref4 = SelfRef(512,512)
        self.selfref3 = SelfRef(256,256)
        self.selfref2 = SelfRef(128,128)
        self.selfref1 = SelfRef(64,64)

        self.ca512=ChannelAttention(512)
        self.sa256=SpatialAttention(256)
        self.sa128=SpatialAttention(128)
        self.sa64=SpatialAttention(64)

        self.refunet = RefUnet(1,64)

    def forward(self, x, phase='test'):
        x_size = x.size()[-2:]
        hx = x

        ## -------------Encoder-------------
        h1, h2, h3, h4, h5 = self.encoder(hx)
        h6=self.conv6(h5)
          
        h1 = self.post1(h1)
        h2 = self.post2(h2)
        h3 = self.post3(h3)
        h4 = self.post4(h4)
        h5 = self.post5(h5)
        h6 = self.post5(h6)

        ## -------------DEM-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))
        atrous_block1=self.atrous_block1(hbg)
        atrous_block3=self.atrous_block3(hbg)
        atrous_block5=self.atrous_block5(hbg)
        atrous_block7=self.atrous_block7(hbg)
        battention=torch.sigmoid(self.gap(hbg))
        battention=battention.expand_as(hbg)
        hbg=torch.cat([atrous_block1, atrous_block3, atrous_block5, atrous_block7, battention], dim=1)
        hbg=self.conv_1x1_output(hbg)

        ## -------------Decoder-------------
        # deep self refinement
        h6 = self.selfref6(h6)
        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx))) # 8 512 5 5
        hx=self.ca512(hx)
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 8 -> 16

        h5 = self.selfref5(h5)
        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hx=self.ca512(hx)
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 16 -> 32

        h4=self.selfref4(h4)
        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hx=self.ca512(hx)
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64

        h3=self.selfref3(h3)
        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hx=self.sa256(hx) 
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 64 -> 128

        h2=self.selfref2(h2)
        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hx=self.sa128(hx)
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2) # 128 -> 256

        h1=self.selfref1(h1)
        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hx=self.sa64(hx)
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = nn.functional.interpolate(db, size=x_size, mode='bilinear')

        d6 = self.outconv6(hd6)
        d6 = nn.functional.interpolate(d6, size=x_size, mode='bilinear')

        d5 = self.outconv5(hd5)
        d5 = nn.functional.interpolate(d5, size=x_size, mode='bilinear')

        d4 = self.outconv4(hd4)
        d4 = nn.functional.interpolate(d4, size=x_size, mode='bilinear')

        d3 = self.outconv3(hd3)
        d3 = nn.functional.interpolate(d3, size=x_size, mode='bilinear')

        d2 = self.outconv2(hd2)
        d2 = nn.functional.interpolate(d2, size=x_size, mode='bilinear')

        d1 = self.outconv1(hd1) # 256
        d1 = nn.functional.interpolate(d1, size=x_size, mode='bilinear')

        ## -------------Refine Module-------------
        dout = self.refunet(d1) # 256

        out_dict = {}
        out_dict['final'] = dout
        out_dict['sal'] = [dout, d1, d2, d3, d4, d5, d6, db]
        return out_dict

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=groups,
                                base_width=self.base_width, dilation=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
