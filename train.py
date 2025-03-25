import yaml
import time
import os
import shutil
from engine import trainer
from models import get_model
import torch
import argparse
import numpy as np
import random
from utils.utils import EarlyStopper, get_dataset, plot_results
from loss import CombinedLoss
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/train.yaml', metavar='DIR', help='configs')

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
SAVE_DIR = config["SAVE_DIR"]
CUDA_DEVICE = int(config["CUDA_DEVICE"])
VAL_PLOT_PER_EPOCH = int(config["VAL_PLOT_PER_EPOCH"])
TEST_PLOT_PER_EPOCH = int(config["TEST_PLOT_PER_EPOCH"])
PLOT_IMAGES = config["PLOT_IMAGES"]
OUTPUT_RANGE = config["OUTPUT_RANGE"]
AUGMENT = str(config["AUGMENT"])
SPLIT_RATIO = config['split']
LOAD_CHECKPOINT = str(config['LOAD_CHECKPOINT'])
STRIDE = int(config['STRIDE'])
ddp = str(config['DDP'])
DROP_LAST = config['DROP_LAST']
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
print(f"Using {DEVICE} device")

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True


def main():
    START_seed()
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    os.mkdir(SAVE_DIR + run_id)
    save_dir = SAVE_DIR + run_id
    shutil.copy(args.config, f'{save_dir}/config.yaml')
    sys.stdout = open(f'{save_dir}/logfile', 'w')
    train_loader, val_loader, test_loader, test_loader, _, _, _ = get_dataset(DATASET, PATHS,AUGMENT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK, SAVE_DIR=save_dir, PLOT_IMAGES=PLOT_IMAGES, split=SPLIT_RATIO, output_range = OUTPUT_RANGE, DDP=ddp, STRIDE=STRIDE, drop_last = DROP_LAST)
    #load model
    model = get_model(MODEL, TASK, PRETRAINED, num_classes=NUM_CLASSES, output_range = OUTPUT_RANGE)
    model.to(DEVICE)
    torch.compile(model)
    
    if LOAD_CHECKPOINT!="":
        print('yes')
        checkpoint = torch.load(LOAD_CHECKPOINT)
        model.load_state_dict(checkpoint['model'])
        
    train_loss = CombinedLoss(losses = TRAIN_LOSS, weights= TRAIN_WEIGHTS_LOSS, device=DEVICE, orange = OUTPUT_RANGE)
    val_loss = CombinedLoss(losses = VAL_LOSS, weights= VAL_WEIGHTS_LOSS, device=DEVICE, orange = OUTPUT_RANGE)

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

    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader = test_loader,
        reconstruct_loader = None,
        train_loss_fn=train_loss,
        val_loss_fn = val_loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_name=LEARNING_SCHEDULER,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        early_stopper=early_stopper,
        val_per_epoch = VAL_PLOT_PER_EPOCH,
        test_per_epoch = TEST_PLOT_PER_EPOCH
    )
    print(results)
    plot_results(results=results, save_dir=save_dir)
if __name__ == "__main__":
    main()