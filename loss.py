import torch.nn as nn
import torch
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils.utils import fft_transform
import warnings
import pdb
import numpy as np
import random

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
START_seed()

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights, device, orange) -> None:
        super().__init__()
        self.losses = losses
        self.weights = weights
        self.mse_loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_w_logits_loss = nn.BCEWithLogitsLoss()
        self.perceptual = lpips.LPIPS(net='vgg').to(device)
        self.mssim_loss = SSIM(data_range=1, size_average=True, channel=1)
        self.device = device
        self.output_range = orange
    def forward(self, inputs, targets):
        final_loss = 0
        for i, loss in enumerate(self.losses):
            try:
                if "FFT" in loss:
                    inputs = fft_transform(inputs.detach().cpu().numpy()).to(self.device).float()
                    targets = fft_transform(targets.detach().cpu().numpy()).to(self.device).float()
            except RuntimeWarning as e:
                print(f"Warning caught: {e}")
                # Place a breakpoint here
                import pdb; pdb.set_trace()  # This will pause execution and allow you to debug
            if "MSE" in loss:
                final_loss += self.weights[i] * self.mse_loss(inputs, targets)
            elif "L1" in loss:
                final_loss += self.weights[i] * self.l1loss(inputs, targets)
            elif "SmoothL1" in loss:
                final_loss += self.weights[i] * self.smooth_l1_loss(inputs, targets)
            elif "CrossEntropy" in loss:
                final_loss += self.weights[i] * self.ce_loss(inputs, targets)
            elif "BCEWithLogits" in loss:
                final_loss += self.weights[i] * self.bce_w_logits_loss(inputs, targets)
            elif "Perceptual" in loss:
                if self.output_range == [-1,1]:
                    final_loss += self.weights[i] * self.perceptual(torch.cat([inputs,inputs,inputs],1), torch.cat([targets,targets,targets],1)).mean()
                else:
                    final_loss += self.weights[i] * self.perceptual(torch.cat([(2*inputs-1),(2*inputs-1),(2*inputs-1)],1), torch.cat([(2*targets - 1),(2*targets - 1),(2*targets - 1)],1)).mean()

            elif "SSIM" in loss:
                if self.output_range == [0,1]:
                    final_loss += self.weights[i] * (1 - self.mssim_loss(inputs, targets))
                else:
                    final_loss += self.weights[i] * (1 - self.mssim_loss((inputs+1)/2, targets))

            else:
                raise Exception("Loss not implemented")
        return final_loss