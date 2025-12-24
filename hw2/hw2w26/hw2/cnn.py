import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # ====== YOUR CODE: ======
        curr_in_channels = in_channels
        
        # Get the activation and pooling classes
        ActivationLayer = ACTIVATIONS[self.activation_type]
        PoolingLayer = POOLINGS[self.pooling_type]

        for i, out_channels in enumerate(self.channels):
            # 1. Add Convolutional Layer
            layers.append(
                nn.Conv2d(curr_in_channels, out_channels, **self.conv_params)
            )
            
            # 2. Add Activation Layer
            layers.append(
                ActivationLayer(**self.activation_params)
            )
            
            # Update channels for the next iteration
            curr_in_channels = out_channels
            
            # 3. Add Pooling Layer every P layers
            # We use (i + 1) because i is 0-indexed.
            if (i + 1) % self.pool_every == 0:
                layers.append(
                    PoolingLayer(**self.pooling_params)
                )
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            # 1. Create a dummy input with the same shape as your expected input images
            # self.in_size is (C, H, W), so we unpack it into (Batch, C, H, W)
            dummy_input = torch.zeros(1, *self.in_size)
            
            # 2. Run the dummy input through the feature extractor
            # We use torch.no_grad() as a best practice to ensure no graph is built
            with torch.no_grad():
                output = self.feature_extractor(dummy_input)
            
            # 3. The number of features is the total number of elements in the output
            # (excluding the batch dimension).
            return output.numel()
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        in_features = self._n_features()
        mlp_dims = [*self.hidden_dims, self.out_classes]

        # Get the actual activation class (e.g., nn.ReLU or nn.LeakyReLU)
        ActivationLayer = ACTIVATIONS[self.activation_type]
        
        nonlins = []
        for _ in range(len(self.hidden_dims)):
            # Instantiate the activation with the params from CNN __init__
            nonlins.append(ActivationLayer(**self.activation_params))
        
        # Add Identity for the final layer
        nonlins.append(nn.Identity())

        mlp = MLP(in_dim=in_features, dims=mlp_dims, nonlins=nonlins)
        return mlp

    def forward(self, x: Tensor):
        # ====== YOUR CODE: ======
        # 1. Extract features using the convolutional layers
        # Input shape: [Batch, C, H, W]
        features = self.feature_extractor(x)
        
        # 2. Flatten the 4D tensor into a 2D tensor
        # We keep the batch dimension (dim 0) and collapse everything else.
        # Shape change: [Batch, Channels, Height, Width] -> [Batch, Features]
        flattened = features.view(features.size(0), -1)
        
        # 3. Pass through the MLP to get class scores (logits)
        out = self.mlp(flattened)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
    # 1. Main Path
        main_layers = []
        curr_in = in_channels
        ActivationLayer = ACTIVATIONS[activation_type]
        
        num_convs = len(channels)
        for i in range(num_convs):
            out_channels = channels[i]
            k_size = kernel_sizes[i]
            
            main_layers.append(
                nn.Conv2d(
                    curr_in, out_channels, kernel_size=k_size, 
                    padding=(k_size - 1) // 2, bias=True
                )
            )
            
            # Pattern: [Conv -> Dropout -> BN -> Act] * (n-1) -> Final Conv
            if i < num_convs - 1:
                if dropout > 0:
                    main_layers.append(nn.Dropout2d(p=dropout))
                if batchnorm:
                    main_layers.append(nn.BatchNorm2d(out_channels))
                main_layers.append(ActivationLayer(**activation_params))
            
            curr_in = out_channels

        self.main_path = nn.Sequential(*main_layers)

        # 2. Shortcut Path
        # We need a 1x1 conv if:
        # - Channels change (in_channels != channels[-1])
        if in_channels != channels[-1]:
            self.shortcut_path = nn.Sequential(
                nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False)
            )
        else:
            # When stride=1 and channels match, this is just an identity wire
            self.shortcut_path = nn.Sequential(nn.Identity())



    def forward(self, x: Tensor):
        main = self.main_path(x)
        shortcut = self.shortcut_path(x)
        out = torch.relu(main + shortcut)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        full_channels = [inner_channels[0]] + list(inner_channels) + [in_out_channels]
        
        # Kernel logic:
        # First conv is 1x1 (entry)
        # Middle convs are inner_kernel_sizes
        # Last conv is 1x1 (exit)
        full_kernels = [1] + list(inner_kernel_sizes) + [1]
        
        super().__init__(
            in_channels=in_out_channels,
            channels=full_channels,
            kernel_sizes=full_kernels,
            **kwargs
        )


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======
        curr_in_channels = in_channels
        
        # P is the number of convolutions per residual block
        P = self.pool_every
        channels_list = self.channels
        N = len(channels_list)

        # Iterate through channels in groups of P
        for i in range(0, N, P):
            # Extract the slice of channels for this block
            group_channels = channels_list[i : i + P]
            num_in_group = len(group_channels)
            
            # Use 3x3 dimension-preserving kernels for all internal convolutions
            group_kernels = [3] * num_in_group
            
            # Logic for choosing between Bottleneck and Standard Residual Block
            # Requirement: Use bottleneck if requested AND if input/output channels match
            if self.bottleneck and curr_in_channels == group_channels[-1] and num_in_group >= 2:
                # However, to match your ResidualBottleneckBlock implementation:
                # inner_channels are the middle layers.
                # Let's adjust to ensure exactly P layers total.
                # Block = 1 (squeeze) + len(inner_channels) + 1 (expand)
                # So len(inner_channels) should be num_in_group - 2
                bottleneck_inners = group_channels[1:-1]
                bottleneck_kernels = group_kernels[1:-1]
                
                layers.append(
                    ResidualBottleneckBlock(
                        in_out_channels=curr_in_channels,
                        inner_channels=bottleneck_inners,
                        inner_kernel_sizes=bottleneck_kernels,
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params
                    )
                )
            else:
                # Standard Residual Block
                layers.append(
                    ResidualBlock(
                        in_channels=curr_in_channels,
                        channels=group_channels,
                        kernel_sizes=group_kernels,
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params
                    )
                )

            # Update input tracker for the next block
            curr_in_channels = group_channels[-1]

            # Apply Pooling after every complete block of P convolutions
            # Only if this wasn't a partial block at the very end
            if (i + P) <= N:
                pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling_layer)

        return nn.Sequential(*layers)

class YourCNN(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        activation_type="lrelu",  # Leaky ReLU often performs better than standard ReLU
        activation_params=dict(negative_slope=0.01),
        **kwargs
    ):
        """
        Your custom CNN implementation.
        """
        # We pass these to the parent CNN class
        super().__init__(
            in_size, 
            out_classes, 
            channels, 
            pool_every, 
            hidden_dims, 
            activation_type=activation_type,
            activation_params=activation_params,
            **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []
        
        curr_in_channels = in_channels
        P = self.pool_every
        N = len(self.channels)

        # Set default pooling params if the dict is empty
        p_params = self.pooling_params if self.pooling_params else {"kernel_size": 3}

        for i in range(0, N, P):
            group_channels = self.channels[i : i + P]
            num_in_group = len(group_channels)
            group_kernels = [3] * num_in_group
            
            layers.append(
                ResidualBlock(
                    in_channels=curr_in_channels,
                    channels=group_channels,
                    kernel_sizes=group_kernels,
                    batchnorm=True,
                    dropout=0.2,
                    activation_type=self.activation_type,
                    activation_params=self.activation_params
                )
            )
            
            curr_in_channels = group_channels[-1]

            if (i + P) <= N:
                PoolingLayer = POOLINGS[self.pooling_type]
                # Use the p_params which now definitely has a kernel_size
                layers.append(PoolingLayer(**p_params))

        return nn.Sequential(*layers)
