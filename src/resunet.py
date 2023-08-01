import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class PSP_Module(nn.Module):
    def __init__(self, in_channels, out_channels, bin):
        super(PSP_Module, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(bin)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()
        x = self.global_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size[2:], mode='bilinear', align_corners=True)
        return x


class PSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSP, self).__init__()
        bins = [1, 2, 3, 6]
        self.psp1 = PSP_Module(in_channels, out_channels, bins[0])
        self.psp2 = PSP_Module(in_channels, out_channels, bins[1])
        self.psp3 = PSP_Module(in_channels, out_channels, bins[2])
        self.psp4 = PSP_Module(in_channels, out_channels, bins[3])

    def forward(self, x):
        x1 = self.psp1(x)
        x2 = self.psp2(x)
        x3 = self.psp3(x)
        x4 = self.psp4(x)
        out = torch.cat([x, x1, x2, x3, x4], dim=1)
        return out


class ResUnet(nn.Module):
    def __init__(self, channel=3, filters=[32,64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.up_residual_conv5 = ResidualConv(filters[0],filters[0],1,1)

        self.output_layer = nn.Sequential(
            ResidualConv(filters[0],1,1,1),
            # nn.LogSoftmax(dim=1),
            nn.Sigmoid()
        )

    def forward(self,m):

        # Encode1
        #stage1
        x1 = self.input_layer(m) + self.input_skip(m)#950*750

        #stage2
        x2 = self.residual_conv_1(x1) #475*375
        #stage3
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.bridge(x4)
        
        # Decode
        x5 = self.upsample_1(x5) #512*20*20
        
        x6 = torch.cat([x5,x4],dim=1) # 768*20*20
        x7 = self.up_residual_conv1(x6) #256*20*20

        x7 = self.upsample_2(x7) #256*40*40

        x8 = torch.cat([x7, x3], dim=1) #384*40*40

        x9 = self.up_residual_conv2(x8) #128*40*40

        x9 = self.upsample_3(x9) #128*80*80
        x10 = torch.cat([x9, x2], dim=1) #192*80*80

        x11 = self.up_residual_conv3(x10) #64*80*80

        x11 = self.upsample_4(x11) #64*160*160
        x12 = torch.cat([x11, x1], dim=1) #96*160*160

        x13 = self.up_residual_conv4(x12) 
        x14 = self.up_residual_conv5(x13)
        output = self.output_layer(x14)

        return output



if __name__ =="__main__":
    model=ResUnet()
    a = torch.rand(1,3,256,256)
    b = model(a)
    print(b.shape)
