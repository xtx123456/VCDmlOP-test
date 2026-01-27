# pot_core/models_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- basic blocks --------
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscale with maxpool then DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.seq(x)

class Up(nn.Module):
    """
    Upscale + concat skip + DoubleConv
    in_ch  : channels of decoder input before upsample
    skip_ch: channels of encoder skip to concat
    out_ch : channels after the two convs
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)                      # [B, out_ch, H*2, W*2]
        # 处理可能的1像素对齐误差（偶发）
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)     # [B, skip_ch + out_ch, H, W]
        return self.conv(x)                 # -> [B, out_ch, H, W]

# -------- U-Net --------
class UNetSeg(nn.Module):
    """
    U-Net for VOC (21 classes), input size multiple of 32 (e.g., 256).
    Enc channels: 64,128,256,512,1024  (bottleneck=1024)
    """
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.inc   = DoubleConv(3,   64)     # 256 -> 256
        self.down1 = Down(64,  128)          # 256 -> 128
        self.down2 = Down(128, 256)          # 128 -> 64
        self.down3 = Down(256, 512)          # 64  -> 32
        self.down4 = Down(512, 1024)         # 32  -> 16 (bottleneck)

        # Up: 1024->512 (concat skip 512) -> 512
        self.up1 = Up(1024, 512, 512)        # 16 -> 32
        self.up2 = Up(512,  256, 256)        # 32 -> 64
        self.up3 = Up(256,  128, 128)        # 64 -> 128
        self.up4 = Up(128,   64,  64)        # 128 -> 256

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)         # 64, 256x256
        x2 = self.down1(x1)      # 128,128x128
        x3 = self.down2(x2)      # 256,64x64
        x4 = self.down3(x3)      # 512,32x32
        x5 = self.down4(x4)      # 1024,16x16

        y  = self.up1(x5, x4)    # 512,32x32
        y  = self.up2(y,  x3)    # 256,64x64
        y  = self.up3(y,  x2)    # 128,128x128
        y  = self.up4(y,  x1)    # 64, 256x256
        logits = self.outc(y)    # C, 256x256
        return logits
