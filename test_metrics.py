from engine import trainer, val_step
from utils.utils import plot_results, plot_metrics
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
from dataset import BrazilDatasetFinetuning, BrazilDatasetPretraining
import shutil, sys

from pathlib import Path


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
DROP_LAST = config['DROP_LAST']

if not os.path.exists(os.path.join(os.path.dirname(LOAD_DIR), SAVE_DIR)):
    os.makedirs(os.path.join(os.path.dirname(LOAD_DIR), SAVE_DIR))

FINAL_DIR = os.path.join(os.path.dirname(LOAD_DIR), SAVE_DIR)
print(FINAL_DIR)
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')
directory = Path(LOAD_DIR)  # Change this to your target directory

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
    shutil.copy(args.config, f'{FINAL_DIR}/config.yaml')
    sys.stdout = open(f'{FINAL_DIR}/logfile', 'w')
    pth_files_sorted = sorted(directory.rglob("checkpoint_*.pth"), key=lambda x: int(x.name.split('_')[1].split('.')[0]))

    final_results = []
    for index,model_path in enumerate(list(pth_files_sorted)):
        model_path = str(model_path)
        LOAD_DIR = model_path
        print(LOAD_DIR)
        if ('best_checkpoint' in LOAD_DIR) or ('last_checkpoint' in LOAD_DIR):
            continue
        metrics = []    
        for path in PATHS:
            print(path)
            val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ])

            test_data  = [path]
            test_dataset = BrazilDatasetFinetuning(datasets=test_data, patch_dim=IMAGE_SIZE, fill_value=0, stride = STRIDE, transform=val_transform)
            print(len(test_dataset))
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last = DROP_LAST)
            model = get_model(MODEL, TASK, PRETRAINED, num_classes=NUM_CLASSES, output_range = OUTPUT_RANGE)

            model.to(DEVICE)
            torch.compile(model)
            

            val_loss = CombinedLoss(losses = VAL_LOSS, weights= VAL_WEIGHTS_LOSS, device=DEVICE, orange=OUTPUT_RANGE)

            #train model
            results = trainer(
                model=model,
                train_loader=test_loader,
                val_loader=test_loader,
                test_loader = test_loader,
                reconstruct_loader = test_loader,
                train_loss_fn=None,
                val_loss_fn = val_loss,
                optimizer=None,
                lr_scheduler=None,
                lr_scheduler_name=LEARNING_SCHEDULER,
                device=DEVICE,
                epochs=NUM_EPOCHS,
                save_dir=LOAD_DIR,
                early_stopper=None,
                val_per_epoch = VAL_PLOT_PER_EPOCH,
                test_per_epoch = TEST_PLOT_PER_EPOCH,
                pads = None,
                test_dataset = test_dataset,
                test_overlap = STRIDE,
                image_size = IMAGE_SIZE,
                final_dir = FINAL_DIR,
                path = path,
                )    
            metrics.append(results[3])
            print(metrics)
        data_np = np.array(metrics)  # Convert list to NumPy array
        averages = np.mean(data_np, axis=0)  # Compute mean along the first axis

        result = averages.tolist()  # Convert back to list
        final_results.append(result)
        print(result)
        plot_metrics(final_results=final_results, save_dir=FINAL_DIR)

if __name__ == "__main__":
    main()