from torch import nn, cat, randn, manual_seed

"""Generator Architecture"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        manual_seed(0)
        self.name = "generator"
        self.outlayer = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)

        # Add all the encoding blocks
        self.down_stack = nn.ModuleList([
            self.downsample(in_filter=3, out_filter=64, kernel_size=4, stride=2, batch_norm=False),
            self.downsample(in_filter=64, out_filter=128, kernel_size=4, stride=2),
            self.downsample(in_filter=128, out_filter=256, kernel_size=4, stride=2),
            self.downsample(in_filter=256, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2),
            self.downsample(in_filter=512, out_filter=512, kernel_size=4, stride=2, batch_norm=False)
        ])
        # Add all the decoding blocks
        self.up_stack = nn.ModuleList([
            self.upsample(in_filter=512, out_filter=512, kernel_size=4, stride=2, dropout=True),
            self.upsample(in_filter=1024, out_filter=512, kernel_size=4, stride=2, dropout=True),
            self.upsample(in_filter=1024, out_filter=512, kernel_size=4, stride=2, dropout=True),
            self.upsample(in_filter=1024, out_filter=512, kernel_size=4, stride=2),
            self.upsample(in_filter=1024, out_filter=256, kernel_size=4, stride=2),
            self.upsample(in_filter=512, out_filter=128, kernel_size=4, stride=2),
            self.upsample(in_filter=256, out_filter=64, kernel_size=4, stride=2)
        ])
    
    def downsample(self, in_filter: int, out_filter: int, kernel_size: int, stride: int, batch_norm: bool=True):
        """
        Image encoder

        Arguments:
            in_filter: the number of channels of the input
            out_filter: the number of channels of the output
            kernel_size: the size of the kernel for convolution
            stride: the value of stride for convolution
            batch_norm: boolean to apply batch normalization

        Return:
            encoder: the nerual network layer(s) used for encoding
        """
        # Find the value for 'same' padding
        padding = int((kernel_size - stride) / 2)

        # Add the steps for the encoder
        encoder = nn.Sequential()
        encoder.append(nn.Conv2d(in_filter, out_filter, kernel_size, stride=stride, padding=padding))
        if batch_norm:
            encoder.append(nn.BatchNorm2d(out_filter))
        encoder.append(nn.LeakyReLU())
        
        return encoder
    
    def upsample(self, in_filter: int, out_filter: int, kernel_size: int, stride: int, dropout: bool=False):
        """
        Image decoder

        Arguments:
            in_filter: the number of channels of the input
            out_filter: the number of channels of the output
            kernel_size: the size of the kernel for convolution
            stride: the value of stride for convolution
            dropout: boolean to apply dropout

        Return:
            decoder: the nerual network layer(s) used for decoding
        """
        # Find the value for 'same' padding
        padding = int((kernel_size - stride) / 2)

        # Add the steps for the decoder
        decoder = nn.Sequential()
        decoder.append(nn.ConvTranspose2d(in_filter, out_filter, kernel_size, stride=stride, padding=padding))
        decoder.append(nn.BatchNorm2d(out_filter))
        if dropout:
            decoder.append(nn.Dropout(p=0.5))
        decoder.append(nn.ReLU())
        
        return decoder

    def forward(self, x):
        """
        Forward propagation for the U-Net generator

        Arguments:
            x: the input image [batch_size, 3, 256, 256]

        Return:
            x: the output of the U-Net generator [batch_size, 3, 256, 256]
        """
        
        # Run the encoders
        skips = []
        for layer in self.down_stack:
            # print(x.shape)
            x = layer(x)
            skips.append(x)   # save the outputs to be used as skip connections later
        
        # Run the decoders
        skips = reversed(skips[:-1])
        for layer, skip in zip(self.up_stack, skips):
            # print(x.shape)
            x = layer(x)
            x = cat([x, skip], dim=1)   # concatinate the result with the skip connections
        
        x = nn.Tanh()(self.outlayer(x))

        return x

"""Discriminator Architecture"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.name = "discriminator"
        manual_seed(0)
        self.padding = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1)
        self.batchnorm = nn.BatchNorm2d(512)
        self.activation = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1)
    
    def downsample(self, in_filter: int, out_filter: int, kernel_size: int, stride: int, batch_norm: bool=True):
        """
        Image encoder

        Arguments:
            in_filter: the number of channels of the input
            out_filter: the number of channels of the output
            kernel_size: the size of the kernel for convolution
            stride: the value of stride for convolution
            batch_norm: boolean to apply batch normalization

        Return:
            encoder: the nerual network layer(s) used for encoding
        """
        # Find the value for 'same' padding
        padding = int((kernel_size - stride) / 2)

        # Add the steps for the encoder
        encoder = nn.Sequential()
        encoder.append(nn.Conv2d(in_filter, out_filter, kernel_size, stride=stride, padding=padding))
        if batch_norm:
            encoder.append(nn.BatchNorm2d(out_filter))
        encoder.append(nn.LeakyReLU())
        
        return encoder
    
    def forward(self, input_image, generated_image):
        """
        Forward propagation for the PatchGAN discriminator

        Arguments:
            input_image: the input image [batch_size, 3, 256, 256]
            generated_image: the generated image from the Generator [batch_size, 3, 256, 256]

        Return:
            x: the output grid for classifying real/fake pixel-by-pixel [batch_size, 1, 30, 30]
        """
        x = cat([input_image, generated_image], dim=1)   # concatenate the two inputs of the function

        # Run the encoders
        x = self.downsample(in_filter=6, out_filter=64, kernel_size=4, stride=2, batch_norm=False)(x)
        x = self.downsample(in_filter=64, out_filter=128, kernel_size=4, stride=2)(x)
        x = self.downsample(in_filter=128, out_filter=256, kernel_size=4, stride=2)(x)

        # Run the final convolutional layers
        x = self.conv1(self.padding(x))
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.conv2(self.padding(x))

        return x

# The following code can be used to debug the dimension of models' outputs
"""
model1 = Generator()
tensor1 = randn(7, 3, 256, 256)
output1 = model1(tensor1)
print(output1.shape)

print(sum(p.numel() for p in model1.parameters()))

model2 = Discriminator()
tensor2 = randn(7, 3, 256, 256)
output2 = model2(output1, tensor2)
print(output2.shape)
"""