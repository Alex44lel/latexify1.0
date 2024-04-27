from torch import nn
from torchvision import models


def convnext_tiny(**kwargs):
    model = models.convnext_tiny(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model


def convnext_small(**kwargs):
    model = models.convnext_small(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model


def convnext_base(**kwargs):
    model = models.convnext_base(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model


def convnext_large(**kwargs):
    model = models.convnext_large(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model
