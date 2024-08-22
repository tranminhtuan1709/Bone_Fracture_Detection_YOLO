import torch
import torch.nn as nn

'''
(kernel_size, out_channels, stride, padding)

[(), (), numbers] means a list contains two tuples and
the number of repeted times.

M represents a Maxpool Layer.
'''

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        '''
            Create a convolutional layer.

            Args:
                in_channels (int): the number of channels of input.
                out_channels (int): the number of channels of output.
                **kwargs: other arguments.

            Returns:
                None
        '''

        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # Normalizes output to the form of mean = 0 and deveiation = 1
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            Forward the input through the layer.
            
            Args:
                x (torch.Tensor): input

            Returns:
                torch.Tensor: a tensor represents the output after applying
                this layer to the given input.
        '''

        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs) -> None:
        '''
            Create yolov1 model with both convolutional and
            fully connected layers.

            Args:
                in_channels (int): the number of channels of input.
                **kwargs: other arguments.

            Returns:
                None
        '''

        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            Forward the input through the entire network.

            Args:
                x (torch.Tensor): input.

            Returns:
                torch.Tensor: a tensor represents the output after applying
                entire network to the given input.
        '''

        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))


    def _create_conv_layers(self, architecture: list) -> nn.Sequential:
        '''
            Create convolutional layers using the initialized architecture.

            Args:
                architecture (list): a list contains tuples represent
                convolutional layers.

            Returns:
                nn.Sequential: a Squential instance.
        '''

        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)


    def _create_fcs(
            self, split_size: int, num_boxes: int, num_classes: int
        ) -> nn.Sequential:
        '''
            Create fully connected layers.

            Args:
                split_size (int): the number of grid cells per row and
                column.
                num_boxes (int): the number of bounding boxes will be
                predicted in each cell.
                num_classes (int): the number of classes will be
                predicted in each cell.

            Returns:
                nn.Sequential: the output.
        '''

        S, B, C = split_size, num_boxes, num_classes
        
        # The final output after forwarding the given input through
        # all fully connected layers has the shape
        # of (batch, S * S * (C + B * 5))
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )
