import torch
import torch.nn as nn
import numpy as np

import einops
from einops import rearrange, reduce
from torch.nn.modules import channelshuffle

class MLPBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout=0.):
        super().__init__()
        
        self.linear1 = nn.Linear(in_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, in_dim)
        
    def forward(self, x):
        y = self.linear1(x)
        y = self.gelu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        return  y

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, tokens_mlp_dim, channels_mlp_dim, dropout=0.):
        super().__init__()

        self.layernorm = nn.LayerNorm(dim)
        self.MLPBlock_token = MLPBlock(num_patches, tokens_mlp_dim, dropout)
        self.MLPBlock_channel = MLPBlock(dim, channels_mlp_dim, dropout)

    def forward(self, x):
        # token mixing
        y = self.layernorm(x)
        y = rearrange(y, 'b p c -> b c p')  # (batch, num_patches, dim(channels)) -> (batch, dim(channels), num_patches)
        y = self.MLPBlock_token(y)
        y_token = rearrange(y, 'b c p -> b p c')  # (batch, dim(channels), num_patches) -> (batch, num_patches, dim(channels)) 

        # channel_mixing
        y = self.layernorm(y_token)
        y = self.MLPBlock_channel(y)
        y_channel = y_token + y
        return y_channel

class MlpMixer(nn.Module):
    def __init__(self, in_channels=3, dim=768, num_classes=10, patch_size=16, image_size=224, depth=8, tokens_mlp_dim=256, channels_mlp_dim=2048):
        super().__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size"
        
        # patch embedding
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        # Mixer Layer
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(
               MixerBlock(dim, self.num_patches, tokens_mlp_dim, channels_mlp_dim)
            )

        # Head 
        self.layernorm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # patch embedding
        y = self.conv(x)
        y = rearrange(y, "n c h w -> n (h w) c")

        # Mixer Layer
        for mixer_block in self.mixer_blocks:
            y = mixer_block(y)
        
        # Head
        y = self.layernorm(y)
        y = y.mean(dim=1)
        y = self.classifier(y)
        return y


if __name__ == "__main__":
    from torchsummary import summary
    import pdb
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # --------- base_model_param ---------
    in_channels = 3
    hidden_size = 768  # channels 512
    num_classes = 1000
    patch_size = 16
    resolution = 224
    number_of_layers = 1
    tokens_mlp_dim = 256
    channels_mlp_dim = 2048
    # ------------------------------------

    model = MlpMixer(
        in_channels=in_channels,
        dim=hidden_size,
        num_classes=num_classes,
        patch_size=patch_size,
        image_size=resolution,
        depth=number_of_layers,
        tokens_mlp_dim=tokens_mlp_dim,
        channels_mlp_dim=channels_mlp_dim
    )
    img = torch.rand(2, 3, 224, 224)

    model = model.to(device)
    img = img.to(device)
    
    output = model(img)
    print(output.shape)

    # summary(model, input_size=(3, 224, 224), device='cpu')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000  # 18.528 million
    print('Trainable Parameters: %.3fM' % parameters)
    summary(model, input_size=(3, 224, 224))  # check model summary
