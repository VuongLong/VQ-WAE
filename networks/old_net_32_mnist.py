from torch import nn
from networks.util import ResBlock


class EncoderVqResnet32(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(EncoderVqResnet32, self).__init__()
        
        # Convolution layers
        layers_conv = []
        layers_conv.append(nn.Conv2d(1, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU(True))
        self.conv = nn.Sequential(*layers_conv)
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        
    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        return mu


class DecoderVqResnet32(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet32, self).__init__()
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        # Convolution layers
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 1, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)

        return out
