from torch import nn, cat, randn

"""Generator Architecture"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.name = "generator"
    
    def downsample(self, in_filter: int, out_filter: int, kernel_size: int, stride: int, batch_norm: bool=True):
        
        padding = int((kernel_size - stride) / 2)

        encoder = nn.Sequential()
        encoder.append(nn.Conv2d(in_filter, out_filter, kernel_size, stride=stride, padding=padding))
        if batch_norm:
            encoder.append(nn.BatchNorm2d(out_filter))
        encoder.append(nn.LeakyReLU())
        
        return encoder
    
    def upsample(self, in_filter: int, out_filter: int, kernel_size: int, stride: int, dropout: bool=False):
        
        padding = int((kernel_size - stride) / 2)

        decoder = nn.Sequential()
        decoder.append(nn.ConvTranspose2d(in_filter, out_filter, kernel_size, stride=stride, padding=padding))
        decoder.append(nn.BatchNorm2d(out_filter))
        if dropout:
            decoder.append(nn.Dropout(p=0.5))
        decoder.append(nn.ReLU())
        
        return decoder

    def forward(self, x):
        
        down_stack = [
            self.downsample(in_filter=3, out_filter=64, kernel_size=4, stride=2, batch_norm=False),
            self.downsample(in_filter=64, out_filter=128, kernel_size=4, stride=2),
            self.downsample(in_filter=128, out_filter=256, kernel_size=4, stride=2),
            self.downsample(in_filter=256, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2, batch_norm=False)
        ]

        up_stack = [
            self.upsample(in_filter=512, out_filter=512, kernel_size=4, stride=2, dropout=True),
            self.upsample(in_filter=1024, out_filter=512, kernel_size=4, stride=2, dropout=True),
            self.upsample(in_filter=1024, out_filter=512, kernel_size=4, stride=2, dropout=True),
            self.upsample(in_filter=1024, out_filter=512, kernel_size=4, stride=2),
            self.upsample(in_filter=1024, out_filter=256, kernel_size=4, stride=2),
            self.upsample(in_filter=512, out_filter=128, kernel_size=4, stride=2),
            self.upsample(in_filter=256, out_filter=64, kernel_size=4, stride=2)
        ]

        outlayer = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)

        skips = []
        for layer in down_stack:
            # print(x.shape)
            x = layer(x)
            skips.append(x)
        print("output: " + str(x.shape))
        skips = reversed(skips[:-1])
        for layer, skip in zip(up_stack, skips):
            # print(x.shape)
            x = layer(x)
            x = cat([x, skip], dim=1)
        
        x = nn.Tanh()(outlayer(x))

        return x

model = Generator()
tensor = randn(1, 3, 256, 256)
output = model(tensor)