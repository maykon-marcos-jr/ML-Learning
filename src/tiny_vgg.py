
def get_out_shape(
        in_shape: tuple[int, int],
        kernel: tuple[int, int],
        stride: tuple[int, int],
        pad: tuple[int, int],
        dilation: tuple[int, int],
    ) -> tuple[int, int]:

    out = [0, 0]
    for i, inp in enumerate(in_shape):
        a = inp + 2*pad[i]
        b = dilation[i]*(kernel[i] - 1)
        c = (a - b - 1) // stride[i]
        out[i] = 1 + c

    return out[0], out[1]

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    # For the Conv2D Layers
    KERNEL_SIZE = (3, 3)
    STRIDE = (1, 1)
    PADDING = (1, 1)
    DILATION = (1, 1)
    ## For The MaxPool2D
    KERNEL_SIZE_M = (2, 2)
    STRIDE_M = (2, 2)
    PADDING_M = (0, 0)
    DILATION_M = (1, 1)
    # each layer of the network compresses and changes
    # the shape of the input data.

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 entry_shape: tuple[int, int] = (64, 64)
        ) -> None:
        super().__init__()

        #### 1º Layer
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE,
                padding=self.PADDING,
                dilation=self.DILATION,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_units,
                hidden_units,
                self.KERNEL_SIZE,
                self.STRIDE,
                self.PADDING,
                self.DILATION,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=self.KERNEL_SIZE_M,
                stride=self.STRIDE_M
            )
        )

        # Conv2D 1
        entry_shape = get_out_shape(
            entry_shape,
            self.KERNEL_SIZE,
            self.STRIDE,
            self.PADDING,
            self.DILATION,
        )
        # Conv2D 2
        entry_shape = get_out_shape(
            entry_shape,
            self.KERNEL_SIZE,
            self.STRIDE,
            self.PADDING,
            self.DILATION,
        )
        # MaxPool2D
        entry_shape = get_out_shape(
            entry_shape,
            self.KERNEL_SIZE_M,
            self.STRIDE_M,
            self.PADDING_M,
            self.DILATION_M,
        )

        #### 2º Layer
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, self.KERNEL_SIZE, self.STRIDE, self.PADDING, self.DILATION,),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, self.KERNEL_SIZE, self.STRIDE, self.PADDING,self.DILATION,),
            nn.ReLU(),
            # default stride value is same as kernel_size
            nn.MaxPool2d(self.KERNEL_SIZE_M, self.STRIDE_M)
        )
        # Conv2D 1
        entry_shape = get_out_shape(
            entry_shape,
            self.KERNEL_SIZE,
            self.STRIDE,
            self.PADDING,
            self.DILATION,
        )
        # Conv2D 2
        entry_shape = get_out_shape(
            entry_shape,
            self.KERNEL_SIZE,
            self.STRIDE,
            self.PADDING,
            self.DILATION,
        )
        # MaxPool2D
        entry_shape = get_out_shape(
            entry_shape,
            self.KERNEL_SIZE_M,
            self.STRIDE_M,
            self.PADDING_M,
            self.DILATION_M,
        )
        flattened = entry_shape[0]*entry_shape[0]
        ### Outer Layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*flattened,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
