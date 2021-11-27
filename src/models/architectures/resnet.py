# Adapted from starter project, ugly. 

import torch
from torch import nn
import torchvision.models as models


class ResNet(nn.Module):
    """
    This is a resnet wrapper class which takes existing resnet architectures and
    adds a final linear layer at the end, ensuring proper output dimensionality
    """

    def __init__(self, model_name, output_dim: int): 
        super().__init__()

        # Use a resnet-style backend
        if "resnet18" == model_name:
            model_func = models.resnet18
        elif "resnet34" == model_name:
            model_func = models.resnet34
        elif "resnet50" == model_name:
            model_func = models.resnet50
        elif "resnet101" == model_name:
            model_func = models.resnet101
        elif "resnet152" == model_name:
            model_func = models.resnet152
        else:
            raise Exception(f"Unknown backend model type: {model_name}")

        # https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        b_model = model_func()
        encoder = nn.Sequential(
            b_model.conv1,
            b_model.bn1,
            b_model.relu,
            b_model.maxpool,
            b_model.layer1,
            b_model.layer2,
            b_model.layer3,
            b_model.layer4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        # Construct the final layer
        feat_dim = b_model.fc.in_features
        final_layer = nn.Sequential(nn.Linear(feat_dim, output_dim))

        self.encoder = encoder
        self.final_layer = final_layer

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return self.final_layer(x)
        
