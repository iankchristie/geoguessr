import torch
import torch.nn as nn
from torchvision import models
from functions import *
from visualization import *
from data_set import *
import pdb


class GeoLocalizationModel(nn.Module):
    def __init__(self, output_size):
        super(GeoLocalizationModel, self).__init__()

        # Use a pretrained model (e.g., ResNet18)
        self.feature_extractor = models.resnet18()

        # Load the downloaded weights
        self.feature_extractor.load_state_dict(torch.load("./resnet18-f37072fd.pth"))

        # Modify the last layer for num_output_neurons outputs (for grid mapping)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, output_size)

    def forward(self, x):
        return self.feature_extractor(x)
