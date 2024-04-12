# -*-coding:utf-8-*-
from __future__ import print_function

from os import environ
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import models.netvlad as netvlad

environ["WANDB_API_KEY"] = ""


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def load_mrNVLAD_model():
    nocuda = False
    num_clusters = 64
    arch = "vgg16-12"
    resume = "./models/weights/"

    input_transform = transforms.Compose([
        # transforms.Resize((wandb.config.imgHeight, wandb.config.imgWidth)),
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    cuda = not nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

    reqGradLayer = int(arch.split("-")[1]) * -1
    encoder = models.vgg16(pretrained=False)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[: -2]  # Layers to be trimmed from end
    encoder_dim = layers[-1].out_channels
    for l in layers[:reqGradLayer]:
        for p in l.parameters():
            p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module("encoder", encoder)

    net_vlad = netvlad.NetVLAD(
        num_clusters=num_clusters, dim=encoder_dim, vladv2=False
    )
    model.add_module("pool", net_vlad)

    num_pcs = 4096
    netvlad_output_dim = encoder_dim
    netvlad_output_dim *= num_clusters
    pca_conv = nn.Conv2d(
        netvlad_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0
    )
    model.add_module(
        "WPCA", nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])
    )
    checkpoint_tar = "checkpoint_WPCA_4096.pth.tar"

    resume_ckpt = join(resume, checkpoint_tar)
    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(
        resume_ckpt, map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["state_dict"])
    assert checkpoint["num_pcs"] == 4096
    model = model.to(device)

    model.eval()

    return {
        "model": model,
        "input_transform": input_transform,
        "device": device
    }


def appear_embed(model, fv, input_transform, device):
    with torch.no_grad():
        pool_size = 4096
        scaleSpace = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        img = Image.fromarray(fv)
        img = input_transform(img)
        model_input = img.unsqueeze(0)

        model_input = model_input.to(device)
        image_encoding = model.encoder(model_input)
        image_encoding = image_encoding.view(
            image_encoding.size(0), image_encoding.size(1), -1)
        for l in scaleSpace:
            input_scaled = model_input[:, :, ::l, ::l]
            image_scaled_encoding = model.encoder(input_scaled)
            image_encoding = torch.cat(
                [
                    image_encoding,
                    image_scaled_encoding.view(
                        image_scaled_encoding.size(0),
                        image_scaled_encoding.size(1),
                        -1,
                    ),
                ],
                dim=2,
            )
        image_encoding = image_encoding.unsqueeze(3)

        vlad_encoding = model.pool(image_encoding)
        vlad_encoding = model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
        dbFeat = np.squeeze(vlad_encoding.detach().cpu().numpy())

    return dbFeat


def appear_embed_stack(model, fv_list, input_transform, device):
    with torch.no_grad():
        pool_size = 4096
        scaleSpace = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        model_input = torch.tensor([])
        for fv_index in range(len(fv_list)):
            fv = fv_list[fv_index]
            img = Image.fromarray(fv)
            img = input_transform(img)
            model_input = torch.cat((model_input, img.unsqueeze(0)), 0)

        model_input = model_input.to(device)
        image_encoding = model.encoder(model_input)
        image_encoding = image_encoding.view(
            image_encoding.size(0), image_encoding.size(1), -1
        )
        for l in scaleSpace:
            input_scaled = model_input[:, :, ::l, ::l]
            image_scaled_encoding = model.encoder(input_scaled)
            image_encoding = torch.cat(
                [
                    image_encoding,
                    image_scaled_encoding.view(
                        image_scaled_encoding.size(0),
                        image_scaled_encoding.size(1),
                        -1,
                    ),
                ],
                dim=2,
            )
        image_encoding = image_encoding.unsqueeze(3)

        vlad_encoding = model.pool(image_encoding)
        vlad_encoding = model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
        dbFeat = np.squeeze(vlad_encoding.detach().cpu().numpy())

    return dbFeat
