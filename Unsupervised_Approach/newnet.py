import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1):
        if p is None: p = (k // 2) * d
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class LateralFuse(nn.Module):
    def __init__(self, in_low, in_high, out):
        super().__init__()
        self.proj_low  = nn.Conv2d(in_low,  out, 1, bias=False)
        self.proj_high = nn.Conv2d(in_high, out, 1, bias=False)
        self.mix       = ConvBNReLU(out * 2, out, k=3)

    def forward(self, low, high):
        if high.shape[-2:] != low.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        l = self.proj_low(low)
        h = self.proj_high(high)
        return self.mix(torch.cat([l, h], dim=1))

class FeatureFusionDense(nn.Module):
    """
    Two-stream encoder with lateral fusions (green in the diagram) and a light
    top-down decoder to return an HxW output.
    """
    def __init__(self, in_ch=5, base=32, out_ch=1):
        super().__init__()
        # LOWER path (detail)
        self.l1 = ConvBNReLU(in_ch, base)                 # 1/1
        self.l2 = ConvBNReLU(base, base)                  # 1/1
        self.l3 = ConvBNReLU(base, base*2, s=2)           # 1/2
        self.l4 = ConvBNReLU(base*2, base*2)              # 1/2

        # UPPER path (semantics)
        self.h1 = ConvBNReLU(in_ch, base, s=2)            # 1/2
        self.h2 = ConvBNReLU(base, base*2, s=2)           # 1/4
        self.h3 = ConvBNReLU(base*2, base*4, d=2)         # 1/4
        self.h4 = ConvBNReLU(base*4, base*4, d=4)         # 1/4

        # Lateral fusions
        self.fuse_a = LateralFuse(base,    base,   base)      # at 1/1
        self.fuse_b = LateralFuse(base*2,  base*2, base*2)    # at 1/2
        self.fuse_c = LateralFuse(base*2,  base*4, base*2)    # at 1/2 vs 1/4

        # Decoder / refinement to input size
        self.dec_mid = ConvBNReLU(base*2, base*2)         # on fused (1/2)
        self.dec_up1 = ConvBNReLU(base*2 + base, base)    # fuse with early (1/1)
        self.out_conv = nn.Conv2d(base, out_ch*4, kernel_size=1)
        self.out2_conv = nn.Conv2d(out_ch*4, out_ch, kernel_size=1)
        
        self.Conv_1x1_v2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        H, W = x.shape[-2:]

        # Lower
        l1 = self.l1(x)       # 1/1
        l2 = self.l2(l1)      # 1/1
        l3 = self.l3(l2)      # 1/2
        l4 = self.l4(l3)      # 1/2

        # Upper
        h1 = self.h1(x)       # 1/2
        f1 = self.fuse_a(l1, h1)        # 1/1 (early fusion for detail)

        h2 = self.h2(h1)      # 1/4
        f2 = self.fuse_b(l3, h2)        # 1/2

        h3 = self.h3(h2)      # 1/4
        h4 = self.h4(h3)      # 1/4
        f3 = self.fuse_c(l4, h4)        # 1/2

        # Combine mid/deep fusions, then decode
        if f2.shape[-2:] != f3.shape[-2:]:
            f2 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        fused_12 = self.dec_mid(f2 + f3)                   # 1/2
        up = F.interpolate(fused_12, size=(H, W), mode="bilinear", align_corners=False)

        # Final refinement with the earliest fused features at full res
        up = torch.cat([up, f1], dim=1)                    # (base*2 -> up) + (base)
        up = self.dec_up1(up)                              # 1/1

        y = self.out_conv(up)      
        y = self.out2_conv(y)                        # (B, out_ch, H, W)
        d11 = self.Maxpool(y)
        d111 = nn.functional.interpolate(d11,size=(70,70))
        
        #dout = self.Conv_1x1_v2(d111)

        #dout = torch.tanh(self.Conv_1x1_v2(d111))
        dout = torch.sigmoid(self.Conv_1x1_v2(d111))
        # Optional task-specific activation outside (sigmoid/softmax), not applied here
        return dout

# quick check
'''
if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128).to("cuda")
    m = FeatureFusionDense(in_ch=3, base=32, out_ch=1).to("cuda")  # e.g., 2-class logits
    summary(m, input_size=(2, 3, 128, 128))
    y = m(x)
    print(x.shape, "->", y.shape)  # torch.Size([2, 3, 128, 160]) -> torch.Size([2, 2, 128, 160])
'''