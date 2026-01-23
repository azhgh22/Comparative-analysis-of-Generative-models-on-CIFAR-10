import torch
import torch.nn as nn

# ---------------------------------------------------
#           Weight Initialization
# ---------------------------------------------------
class WeightsInitializer:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, module):
        classname = module.__class__.__name__

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, mean=self.mean, std=self.std)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, mean=1.0, std=self.std)
            nn.init.zeros_(module.bias.data)