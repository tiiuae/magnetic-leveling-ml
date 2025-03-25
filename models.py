
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch
import torch.nn.functional as F
import timm
import numpy as np
import random

from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights

from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights

from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

from torchvision.models import squeezenet1_1
from torchvision.models import SqueezeNet1_1_Weights

from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights

from torchvision.models import densenet121
from torchvision.models import DenseNet121_Weights

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

import utils.utils as utils
import clip


def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

START_seed()



class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    



class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=False, residual=False) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.relu1 = nn.ReLU()

        if batch_norm:
            self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.relu2 = nn.ReLU()

        if batch_norm:
            self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)

        self.batch_norm = batch_norm

        self.residual = residual
        self.diff_channels = False

        if out_channels != in_channels:
            self.residual = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.batch_norm_1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.batch_norm_2(out)
        
        if self.residual:
            out = torch.add(out, residual)

        out = self.relu2(out)

        return out

class CNN_PRO(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 cnn_size=[8, 16, 32], cnn_kernels=[7, 5, 3], avg_pool_size=8,
                 fc_size=[1024, 512, 128], 
                 batch_norm=False, dropout=False, residual=False) -> None:
        super().__init__()

        assert len(cnn_size) == len(cnn_kernels)

        feature_extractor_list = []

        for i in range(len(cnn_size)):
            if i == 0:
                feature_extractor_list.append(CNN_block(in_channels=in_channels, out_channels=cnn_size[i], 
                                                        kernel_size=cnn_kernels[i], batch_norm=batch_norm, residual=residual))
            else:
                feature_extractor_list.append(CNN_block(in_channels=cnn_size[i-1], out_channels=cnn_size[i], 
                                                        kernel_size=cnn_kernels[i], batch_norm=batch_norm, residual=residual))
            if i != len(cnn_size)-1:
                feature_extractor_list.append(nn.MaxPool2d(kernel_size=2))

        feature_extractor_list.append(nn.AdaptiveAvgPool2d(output_size=avg_pool_size))

        self.feature_extractor = nn.Sequential(*feature_extractor_list)


        classifier_list = []
        for i in range(len(fc_size)):
            if i == 0:
                classifier_list.append(nn.Linear(in_features=cnn_size[-1]*avg_pool_size*avg_pool_size, out_features=fc_size[i]))
            else:
                classifier_list.append(nn.Linear(in_features=fc_size[i-1], out_features=fc_size[i]))
 
            
            if batch_norm:
                    classifier_list.append(nn.BatchNorm1d(num_features=fc_size[i]))

            classifier_list.append(nn.ReLU())

            if dropout:
                classifier_list.append(nn.Dropout(p=0.5))

        classifier_list.append(nn.Linear(in_features=fc_size[-1], out_features=num_classes))

        self.classifier = nn.Sequential(*classifier_list)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)
        out = self.classifier(features)
        return out

    def num_of_params(self):
        total = 0
        for layer_params in self.feature_extractor.parameters():
            total += layer_params.numel()
        for layer_params in self.classifier.parameters():
            total += layer_params.numel()
        return total   
    

class MLP(nn.Module):
    def __init__(self, num_classes=1, in_features=3*224*224, fc_size=[4096, 2048, 1024, 512, 256, 128], 
                 batch_norm=True, dropout=True) -> None:
        super().__init__()

        classifier_list = []
        for i in range(len(fc_size)):
            if i == 0:
                classifier_list.append(nn.Linear(in_features=in_features, out_features=fc_size[i]))
            else:
                classifier_list.append(nn.Linear(in_features=fc_size[i-1], out_features=fc_size[i]))
 
            
            if batch_norm:
                    classifier_list.append(nn.BatchNorm1d(num_features=fc_size[i]))

            classifier_list.append(nn.ReLU())

            if dropout:
                classifier_list.append(nn.Dropout(p=0.5))

        classifier_list.append(nn.Linear(in_features=fc_size[-1], out_features=num_classes))

        self.classifier = nn.Sequential(*classifier_list)
        
    def forward(self, x):
        features = torch.flatten(x, start_dim=1)
        out = self.classifier(features)
        return out

    def num_of_params(self):
        total = 0
        for layer_params in self.classifier.parameters():
            total += layer_params.numel()
        return total


def get_vgg16(pretrained=False):
    if pretrained:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[26].parameters():
            param.requires_grad = True

        for param in model.features[28].parameters():
            param.requires_grad = True
        
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = True

    else:
        model = vgg16()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.classifier.add_module("7", torch.nn.Linear(1000, 1))
    
    return model


def get_vgg16_cnn(pretrained=False):
    if pretrained:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[26].parameters():
            param.requires_grad = True

        for param in model.features[28].parameters():
            param.requires_grad = True
        
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = True

    else:
        model = vgg16()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=25088,
            out_features=1,
            bias=True
        )
    )
    
    return model


def get_MLP(pretrained=False):
    model = MLP()
    
    return model

def get_resnet50(task, pretrained=False,num_classes=10):
    if task=='Classification':
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = resnet50()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.fc = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=2048,
                out_features=num_classes,
                bias=True
            )
        )
    else:
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = resnet50()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.fc = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=2048,out_features=1),
            # torch.nn.BatchNorm1d(num_features=128),
            # torch.nn.ELU(),
            # torch.nn.Linear(in_features=128,out_features=1),
            # torch.nn.Hardtanh(min_val=0.,max_val=4.)
        )

    
    return model



def get_resnet18(pretrained=False, num_classes=10):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        # for param in model.layer4.parameters():
        #     param.requires_grad = True

        # for i, (name, layer) in enumerate(model.layer4.named_modules()):
        #     if isinstance(layer, torch.nn.Conv2d):
        #         layer.reset_parameters()

    else:
        model = resnet18()

    # Add on fully connected layers for the output of our model

    # model.fc = torch.nn.Sequential(
    #     nn.Dropout(0.2),
    #     torch.nn.Linear(
    #         in_features=512,
    #         out_features=num_classes,
    #         bias=True
    #     ) 
    # )

    model.fc = torch.nn.Linear(
            in_features=512,
            out_features=num_classes
    ) 

    return model

def get_resnet101(pretrained=False):
    if pretrained:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for i, (name, layer) in enumerate(model.layer4.named_modules()):
            if isinstance(layer, torch.nn.Conv2d):
                layer.reset_parameters()

    else:
        model = resnet101()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=2048,
            out_features=1,
            bias=True
        )
    )
    
    return model


def get_mobilenet_v3_small(pretrained=False):
    if pretrained:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(10):
            for param in model.features[i].parameters():
                param.requires_grad = False

    else:
        model = mobilenet_v3_small()

    # Add on fully connected layers for the output of our model

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=576,
            out_features=1,
            bias=True
        )
    )

    return model


def get_mobilenet_v3_large(pretrained=False):
    if pretrained:
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for i in range(13):
            for param in model.features[i].parameters():
                param.requires_grad = False

        for i in range(13, 17):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = mobilenet_v3_large()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=960,
            out_features=512,
            bias=True
        ),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=512,
            out_features=1,
            bias=True
        )
    )

    return model

def get_efficientnet_v2_s(pretrained=False):
    if pretrained:
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(6):
            for param in model.features[i].parameters():
                param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = efficientnet_v2_s()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.2),
        torch.nn.Linear(
            in_features=1280,
            out_features=1,
            bias=True
        )
    )
    return model

def get_squeezenet1_1(pretrained=False):
    if pretrained:
        model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(10):
            for param in model.features[i].parameters():
                param.requires_grad = False

    else:
        model = squeezenet1_1()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.5),
        torch.nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=1,
            stride=1
        ),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    )

    return model


def get_convnext_tiny(pretrained=False):
    if pretrained:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(6):
            for param in model.features[i].parameters():
                param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = convnext_tiny()

    def _is_contiguous(tensor: torch.Tensor) -> bool:
        # jit is oh so lovely :/
        # if torch.jit.is_tracing():
        #     return True
        if torch.jit.is_scripting():
            return tensor.is_contiguous()
        else:
            return tensor.is_contiguous(memory_format=torch.contiguous_format)

    class LayerNorm2d(nn.LayerNorm):
        r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
        """

        def __init__(self, normalized_shape, eps=1e-6):
            super().__init__(normalized_shape, eps=eps)

        def forward(self, x) -> torch.Tensor:
            if _is_contiguous(x):
                return F.layer_norm(
                    x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
            else:
                s, u = torch.var_mean(x, dim=1, keepdim=True)
                x = (x - u) * torch.rsqrt(s + self.eps)
                x = x * self.weight[:, None, None] + self.bias[:, None, None]
                return x

    model.classifier = torch.nn.Sequential(
        LayerNorm2d(768, eps=1e-06),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(
            in_features=768,
            out_features=1,
            bias=True
        )
    )

    return model
    
def get_deitB(task, pretrained=False,num_classes=10):
    if task == 'Classification':
        if pretrained:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=True)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=False)
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.head = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=model.head.in_features,
                out_features=num_classes,
                bias=True
            )
        )
    else:
        if pretrained:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=True)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = timm.create_model('deit_base_patch16_224.fb_in1k',pretrained=False)
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.head = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=model.head.in_features,
                out_features=1,
                bias=True
            )
        )

    return model


def get_medical_densnet121(pretrained=False,num_classes=2):
    if pretrained:
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
        # for param in model.layer4.parameters():
        #     param.requires_grad = True

        # for i, (name, layer) in enumerate(model.layer4.named_modules()):
        #     if isinstance(layer, torch.nn.Conv2d):
        #         layer.reset_parameters()

    else:
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.classifier = torch.nn.Sequential(torch.nn.Linear(
            in_features=1024,
            out_features=num_classes,
            bias=True
        ))
    return model


def get_densnet121(task, pretrained=False,num_classes=2):
    if task == 'Classification':
        if pretrained:
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = densenet121()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=1024,
                out_features=num_classes,
                bias=True
            )
        )
        
    else:
        if pretrained:
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # Freeze model weights
            for param in model.parameters():
                param.requires_grad = False

            # for param in model.layer4.parameters():
            #     param.requires_grad = True

            # for i, (name, layer) in enumerate(model.layer4.named_modules()):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.reset_parameters()

        else:
            model = densenet121()
        

        # Add on fully connected layers for the output of our model

        # model.avgpool = torch.nn.Identity()

        model.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                in_features=1024,
                out_features=1,
                bias=True
            )
        )
        
    return model



def get_CLIP(classnames, pretrained=False, num_classes=10):
    classnames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    base_model, preprocess = clip.load('ViT-B/32', 'cuda', jit = False)
    template = utils.openai_imagenet_template
    clf = utils.zeroshot_classifier(base_model, classnames, template, 'cuda')
    feature_dim = base_model.visual.output_dim
    model = utils.ModelWrapper(base_model, feature_dim, num_classes, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    return model





class SegmentationRefinementNetwork(nn.Module):
    def __init__(self):
        super(SegmentationRefinementNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x





class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, output_range = None, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.output_range = output_range
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.dropout = nn.Dropout(0.2)
        # self.downsample3x = nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        # x1 = self.dropout(x1)
        x2 = self.down1(x1)
        # x2 = self.dropout(x2)

        x3 = self.down2(x2)
        # x3 = self.dropout(x3)

        x4 = self.down3(x3)
        # x4 = self.dropout(x4)


        x5 = self.down4(x4)
        # x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        # x = self.dropout(x)

        x = self.up2(x, x3)
        # x = self.dropout(x)

        x = self.up3(x, x2)
        # x = self.dropout(x)


        x = self.up4(x, x1)

        logits = self.outc(x)
        # logits = self.downsample3x(logits)

        if self.output_range == [-1,1]:
            return torch.tanh(logits)
        else:
            return torch.sigmoid(logits)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)











class DownsampleConv(nn.Module):
    """Learnable Conv layer to downsample skip connections"""
    def __init__(self, in_channels, out_channels):
        super(DownsampleConv, self).__init__()
        self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=3, padding=1)

    def forward(self, x):
        return self.downsample_conv(x)




class UNet_v2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,  output_range = None):
        super(UNet_v2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.output_range = output_range

        # Define layers
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # Upsampling path
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        # Learnable Conv Layers in Skip Connections for Dimension Matching
        self.skip_conv1 = DownsampleConv(64, 64)
        self.skip_conv2 = DownsampleConv(128, 128)
        self.skip_conv3 = DownsampleConv(256, 256)
        self.skip_conv4 = DownsampleConv(512, 512)

        # Output layer to create 128x128 output
        self.outc = OutConv(64, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        # print('x1', x1.shape)
        x2 = self.down1(x1)
        # print('x2', x2.shape)
        x3 = self.down2(x2)
        # print('x3', x3.shape)
        x4 = self.down3(x3)
        # print('x4', x4.shape)
        x5 = self.down4(x4)
        # print('x5', x5.shape)
        # print('conv shape', self.skip_conv4(x4).shape)
        # Decoder with skip connections and dimension matching
        x = self.up1(x5, self.skip_conv4(x4))
        # print('xup1', x.shape)
        # print('conv shape', self.skip_conv3(x3).shape)

        x = self.up2(x, self.skip_conv3(x3))
        # print('xup2', x.shape)
        x = self.up3(x, self.skip_conv2(x2))
        # print('xup3', x.shape)
        x = self.up4(x, self.skip_conv1(x1))
        # print('xup4', x.shape)

        # Final output with center crop to 128x128
        logits = self.outc(x)
        
        if self.output_range == [-1,1]:
            return torch.tanh(logits)
        else:
            return torch.sigmoid(logits)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)




class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ResNetEncoder(torch.nn.Module):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 input_ch=3,
                 z_dim=10,
                 bUseMultiResSkips=True):

        super(ResNetEncoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_ch, out_channels=8,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(n_filters_1, n_filters_2,
                                    kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_2),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                        kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(self.max_filters),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x


class ResNetDecoder(torch.nn.Module):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 output_channels=3,
                 bUseMultiResSkips=True):

        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=z_dim, out_channels=self.max_filters,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(self.max_filters),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 3)
            n_filters_1 = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_filters_0, n_filters_1,
                                             kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_1),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                                 kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(n_filters_1),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, z):

        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)

        return z


class ResNetAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)

        self.fc1 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc2 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h.view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim))

    def decode(self, z):
        h = self.decoder(self.fc2(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        return torch.tanh(h)

    def forward(self, x):
        return self.decode(self.encode(x))


class ResNetVAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetVAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)

        # Assumes the input to be of shape 256x256
        self.fc21 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc22 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc3 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h1 = self.encoder(x).view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.decoder(self.fc3(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)#, mu, logvar


class VectorQuantizer(torch.nn.Module):
    """
    Implementation of VectorQuantizer Layer from: simplegan.autoencoder.vq_vae
    url: https://simplegan.readthedocs.io/en/latest/_modules/simplegan/autoencoder/vq_vae.html
    """
    def __init__(self, num_embeddings, embedding_dim, commiment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commiment_cost = commiment_cost

        self.embedding = torch.nn.parameter.Parameter(torch.tensor(
            torch.randn(self.embedding_dim, self.num_embeddings)),
            requires_grad=True)

    def forward(self, x):

        flat_x = x.view([-1, self.embedding_dim])

        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_x, self.embedding)
            + torch.sum(self.embedding ** 2, dim=0, keepdim=True)
        )

        encoding_indices = torch.argmax(-distances, dim=1)
        encodings = (torch.eye(self.num_embeddings)[encoding_indices]).to(x.device)
        encoding_indices = torch.reshape(encoding_indices, x.shape[:1] + x.shape[2:])
        quantized = torch.matmul(encodings, torch.transpose(self.embedding, 0, 1))
        quantized = torch.reshape(quantized, x.shape)

        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)

        loss = q_latent_loss + self.commiment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return loss, quantized, perplexity, encoding_indices

    def quantize_encoding(self, x):
        encoding_indices = torch.flatten(x)
        encodings = torch.eye(self.num_embeddings)[encoding_indices]
        quantized = torch.matmul(encodings, torch.transpose(self.embedding, 0, 1))
        quantized = torch.reshape(quantized, torch.Size([-1, self.embedding_dim]) + x.shape[1:])
        return quantized


class ResNetVQVAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(3, 256, 256),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 vq_num_embeddings=512,
                 vq_embedding_dim=64,
                 vq_commiment_cost=0.25,
                 bUseMultiResSkips=True):
        super(ResNetVQVAE, self).__init__()

        assert input_shape[1] == input_shape[2]
        image_channels = input_shape[0]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=self.z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=vq_embedding_dim, bUseMultiResSkips=bUseMultiResSkips)

        self.vq_vae = VectorQuantizer(num_embeddings=vq_num_embeddings,
                                      embedding_dim=vq_embedding_dim,
                                      commiment_cost=vq_commiment_cost)
        self.pre_vq_conv = torch.nn.Conv2d(in_channels=self.z_dim, out_channels=vq_embedding_dim,
                                           kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        loss, quantized, perplexity, encodings = self.vq_vae(x)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity, encodings



# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 128 x 128
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
            # State size: 1 x 1 x 1
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output















def get_UNet(task = 'Segmentation', num_classes=1, n_channels=1, output_range = None):
    return UNet(n_channels = n_channels, n_classes = num_classes, output_range=output_range)

def get_UNet_v2(task = 'Segmentation', num_classes=1, n_channels=1, output_range = None):
    return UNet_v2(n_channels = n_channels, n_classes = num_classes, output_range=output_range)


def get_ResNetAutoencoder(task = 'Reconstruction', num_classes=1, n_channels=1):
    return ResNetAE(input_shape=(256,256,1), n_ResidualBlock=2, n_levels=4 , bottleneck_dim = 1024)

def get_ResNetVAE(task = 'Reconstruction', num_classes=1, n_channels=1):
    return ResNetVAE(input_shape=(256,256,1))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

def get_AttU_Net():
    return AttU_Net()


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


def get_R2AttU_Net():
    return R2AttU_Net()


def get_model(model_name, task = 'Classification', pretrained=False, num_classes=10, classnames = None, output_range = None):
    if model_name == "ResNet50":
        return get_resnet50(task, pretrained, num_classes)
    elif model_name == "ResNet18":
        return get_resnet18(pretrained, num_classes)
    elif model_name == "ResNet101":
        return get_resnet101(pretrained)
    elif model_name == "MobileNetV3Small":
        return get_mobilenet_v3_small(pretrained)
    elif model_name == "MobileNetV3Large":
        return get_mobilenet_v3_large(pretrained)
    elif model_name == "EfficientNetV2S":
        return get_efficientnet_v2_s(pretrained)
    elif model_name == "SqueezeNet1_1":
        return get_squeezenet1_1(pretrained)
    elif model_name == "ConvNeXtTiny":
        return get_convnext_tiny(pretrained)
    elif model_name == "DeiT-B":
        return get_deitB(task, pretrained, num_classes)
    elif model_name == "MIMIC-DenseNet121":
        return get_medical_densnet121(pretrained, num_classes)
    elif model_name == "DenseNet121":
        return get_densnet121(task, pretrained, num_classes)
    elif model_name == "CLIP":
        return get_CLIP(classnames, pretrained, num_classes)
    elif model_name == "LeNet5":
        return LeNet5(2)
    elif model_name == "CNN_PRO":
        return CNN_PRO(cnn_size=[16, 16, 32, 32, 32, 32], cnn_kernels=[7, 5, 5, 3, 3, 3], 
                        avg_pool_size=8, fc_size=[1024, 512, 256, 128], 
                        dropout=True, batch_norm=True)
    elif model_name == "VGG16":
        return get_vgg16(pretrained)
    elif model_name == "VGG16_CNN":
        return get_vgg16_cnn(pretrained)
    elif model_name == "MLP":
        return get_MLP(pretrained)

    elif model_name == 'UNet':
        return get_UNet(task, num_classes, n_channels = 1, output_range = output_range)
    
    elif model_name == 'UNet_v2':
        return get_UNet_v2(task, num_classes, n_channels = 1, output_range = output_range)

    elif model_name == 'ResNetAutoencoder':
        return get_ResNetAutoencoder(task, num_classes, n_channels = 1)

    elif model_name == 'ResNetVAE':
        return get_ResNetVAE(task, num_classes, n_channels = 1)
    elif model_name =="R2AttU_Net":
        return get_R2AttU_Net()
    elif model_name == "AttU_Net":
        return get_AttU_Net()
    else:
        raise Exception("Model not implemented")
    


