# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class MaskFormerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.mask_features = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.predictor = MultiScaleMaskedTransformerDecoder()

    def forward(self, features, mask=None):
        return self.layers(features,  mask)

    def layers(self, features, mask=None):
        features.reverse()  # reverse the order to top-bottom
        features[0] = self.conv1(features[0])
        features[1] = self.conv2(features[1])
        features[2] = self.conv3(features[2])
        features[3] = self.conv4(features[3])
        features[4] = self.conv5(features[4])
        # feature = self.mask_features(features[-1])
        # multi_scale_features = []
        # multi_scale_features.append(feature)
        multi_scale_features, mask_features = features[:-1], self.mask_features(features[-1])
        # mask_features = feature
        predictions = self.predictor(multi_scale_features, mask_features=mask_features, mask=mask)
        return predictions
