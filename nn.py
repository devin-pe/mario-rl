from torch import nn


class DeepNeuralNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        channels = input_dim[0]

        # Following the conventions that:
        # - convolutional layers at the start of a network increase in size
        # - fully-connected layers at the end of a network decrease in size

        # too small?
        # self.neural_net = nn.Sequential(
        #     nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=7, stride=3),
        #     nn.Mish(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
        #     nn.Mish(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
        #     nn.Mish(),
        #     nn.Flatten(),
        #     nn.Linear(2592, 512),
        #     nn.Mish(),
        #     nn.Linear(512, output_dim)
        # )

        # better, but got stuck at a reward of ~900 on average
        # self.neural_net = nn.Sequential(
        #     nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=7, stride=3),
        #     nn.Mish(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
        #     nn.Mish(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #     nn.Mish(),
        #     nn.Flatten(),
        #     nn.Linear(5184, 512),
        #     nn.Mish(),
        #     nn.Linear(512, output_dim)
        # )
        
        # very good performance on RIGHT_ONLY
        self.neural_net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=8, stride=4),
            nn.Mish(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.Mish(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(8192, 2048),
            nn.Mish(),
            nn.Linear(2048, 512),
            nn.Mish(),
            nn.Linear(512, output_dim)
        )

        
    # forward pass definition
    def forward(self, state):
        return self.neural_net(state)
