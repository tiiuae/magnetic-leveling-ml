from engine import trainer, val_step
from utils.utils import plot_results
from models import get_model
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import argparse
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data.dataset import Subset
from utils.utils import EarlyStopper, get_dataset, postprocess
from loss import CombinedLoss
import yaml
import json
import time
import os
from utils.utils import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/test.yaml', metavar='DIR', help='configs')

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
print(config)

LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
NUM_CLASSES = int(config["NUM_CLASSES"])
PATIENCE = int(config["PATIENCE"])
TRAIN_LOSS = config["TRAIN_LOSS"]
TRAIN_WEIGHTS_LOSS = config["TRAIN_WEIGHTS_LOSS"]
VAL_LOSS = config["VAL_LOSS"]
VAL_WEIGHTS_LOSS = config["VAL_WEIGHTS_LOSS"]
IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]
NUM_WORKERS = int(config["NUM_WORKERS"])
DATASET = config["DATASET"]
TASK = config["TASK"]
PATHS = config["PATH"]
PRETRAINING = config["PRETRAINED"]
LOAD_DIR = config["LOAD_DIR"]
SAVE_DIR = config["SAVE_DIR"]
RECONSTRUCT_OPTION = str(config["RECONSTRUCT_OPTION"])
CREATE_CSV = config["CREATE_CSV"]
CUDA_DEVICE = int(config["CUDA_DEVICE"])
VAL_PLOT_PER_EPOCH = int(config["VAL_PLOT_PER_EPOCH"])
TEST_PLOT_PER_EPOCH = int(config["TEST_PLOT_PER_EPOCH"])
PLOT_IMAGES = config["PLOT_IMAGES"]
OUTPUT_RANGE = config["OUTPUT_RANGE"]
SPLIT = config["split"]
# TEST_OVERLAP = int(config["test_overlap"])
STRIDE = int(config['STRIDE'])

if not os.path.exists(os.path.join(os.path.dirname(LOAD_DIR), SAVE_DIR)):
    os.makedirs(os.path.join(os.path.dirname(LOAD_DIR), SAVE_DIR))

FINAL_DIR = os.path.join(os.path.dirname(LOAD_DIR), SAVE_DIR)
print(FINAL_DIR)
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')

print(f"Using {DEVICE} device")

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()
    _, _, _, reconstruct_loader, train_dataset, val_dataset, test_dataset = get_dataset(DATASET, PATHS, "Minimal", IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK, SAVE_DIR, PLOT_IMAGES, SPLIT, RECONSTRUCT_OPTION, LOAD_DIR=LOAD_DIR, output_range = OUTPUT_RANGE, STRIDE = STRIDE)
    
    #load model
    model = get_model(MODEL, TASK, PRETRAINED, num_classes=NUM_CLASSES, output_range = OUTPUT_RANGE)

    model.to(DEVICE)
    torch.compile(model)
    

    val_loss = CombinedLoss(losses = VAL_LOSS, weights= VAL_WEIGHTS_LOSS, device=DEVICE, orange=OUTPUT_RANGE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, verbose=True)
    elif LEARNING_SCHEDULER == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif LEARNING_SCHEDULER == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    elif LEARNING_SCHEDULER == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1)
    else:
        lr_scheduler = None

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.001)

    #train model
    results = trainer(
        model=model,
        train_loader=_,
        val_loader=_,
        test_loader = _,
        reconstruct_loader = reconstruct_loader,
        train_loss_fn=None,
        val_loss_fn = val_loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_name=LEARNING_SCHEDULER,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=LOAD_DIR,
        early_stopper=early_stopper,
        val_per_epoch = VAL_PLOT_PER_EPOCH,
        test_per_epoch = TEST_PLOT_PER_EPOCH,
        pads = None,
        test_dataset = test_dataset,
        test_overlap = STRIDE,
        image_size = IMAGE_SIZE
        )
    postprocess(results[0], _, FINAL_DIR, PATHS, csv=CREATE_CSV)
if __name__ == "__main__":
    main()


