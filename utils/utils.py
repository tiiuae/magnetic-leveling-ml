import torch
import clip
import pandas as pd
import os
import math
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.integrate import quad
import cv2
import random
from sklearn.metrics import confusion_matrix
import torchvision
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
from timm.data.transforms import RandomResizedCropAndInterpolation
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from dataset import BrazilDatasetPretraining, BrazilDatasetFinetuning
from torchvision import transforms, utils
import torchvision.transforms as T
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

START_seed()


def get_dataset(DATASET, paths, augment, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK,  SAVE_DIR, PLOT_IMAGES, split = [0.6,0.2,0.2], reconstruct_grid='entire', world_size = None, rank = None, DDP=None, LOAD_DIR=None, output_range = None, STRIDE = 32, drop_last = True):
    print(STRIDE)
    if LOAD_DIR!=None:
        SAVE_DIR = os.path.join(LOAD_DIR, SAVE_DIR)
    if output_range == [-1,1]:
        fill = -1
    else:
        fill = 0
    if augment == "Minimal":
        train_transform = v2.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ])

    elif augment == "Medium":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
        ])

    elif augment == "Heavy":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 180), fill = fill),  # Random rotation between 0 and 90 degrees
            
            transforms.RandomAffine(
                degrees=20,  # Random rotation up to 30 degrees
                translate=(0.1, 0.1),  # Random translation up to 30% in both x and y directions
                scale=(0.9, 1.1),  # Random scaling between 80% and 120%
                fill=0  # Fill the area outside the transformed image
            ),
            transforms.ColorJitter(
                brightness=0.2,  # Adjust brightness
                contrast=0.2,    # Adjust contrast
                saturation=0.2,  # Adjust saturation
                hue=0.1          # Adjust hue (if needed, otherwise set to 0)
            ),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),  # Apply Gaussian Blur
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])


    if DATASET == "BRAZILPretrain":
        print('IT IS PRETRAININGGGGGGGGGGGGGGGGGG')

        train_data = [path for path in paths if 'train' in path]
        val_data  = [path for path in paths if ('val1' in path or 'val2' in path)]
        test_data = [path for path in paths if ('train' not in path) and ('val1' not in path) and ('val2' not in path)]
        print(train_data)
        print(val_data)
        print(test_data)
        train_dataset = BrazilDatasetPretraining(datasets=train_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = 24, transform=train_transform)
        val_dataset = BrazilDatasetPretraining(datasets=val_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = STRIDE, transform=val_transform)
        test_dataset = BrazilDatasetPretraining(datasets=test_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = STRIDE, transform=val_transform, split = 'test')
        # if DDP==False:
        #     sampler_train = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
        #     sampler_val = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank)
        #     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler = sampler_train, shuffle=True, num_workers=NUM_WORKERS, drop_last = True)
        #     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler = sampler_val, shuffle=False, num_workers=NUM_WORKERS, drop_last = True)
        #     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler = sampler_val, shuffle=False, num_workers=NUM_WORKERS, drop_last = True)
        
            # print(len(train_loader), len(val_loader), len(test_loader))
            # breakpoint()
            # return train_loader, val_loader, test_loader, test_loader,0, 0 # extremes, pads


    elif DATASET == "BRAZILFinetune":
        print('FINETUNINGGGGGGGGGGGGGGGGG IT IS')

        train_data = [path for path in paths if 'train' in path]
        val_data  = [path for path in paths if ('val1' in path or 'val2' in path)]
        test_data = [path for path in paths if ('train' not in path) and ('val1' not in path) and ('val2' not in path)]
        print(train_data)
        print(val_data)
        print(test_data)

        train_dataset = BrazilDatasetFinetuning(datasets=train_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = 24, transform=train_transform)
        val_dataset = BrazilDatasetFinetuning(datasets=val_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = STRIDE, transform=val_transform)
        test_dataset = BrazilDatasetFinetuning(datasets=test_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = STRIDE, transform=val_transform, split = 'test')
        # if DDP==False:
        #     sampler_train = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
        #     sampler_val = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank)
        #     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler = sampler_train, shuffle=True, num_workers=NUM_WORKERS, drop_last = True)
        #     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler = sampler_val, shuffle=False, num_workers=NUM_WORKERS, drop_last = True)
        #     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler = sampler_val, shuffle=False, num_workers=NUM_WORKERS, drop_last = True)
        
            # print(len(train_loader), len(val_loader), len(test_loader))
            # breakpoint()
            # return train_loader, val_loader, test_loader, test_loader,0, 0 # extremes, pads




    

    elif DATASET == "BRAZIL_v2":
        train_data = paths[:-6]
        val_data  = paths[-6:-1]
        test_data = [paths[-1]]
        print(train_data)
        print(val_data)
        print(test_data)
        # data1,data2 = np.load('train1_crop_final.npy'), np.load('train2_crop_final.npy')
        # noisy_image1, clean_image1 = csv_to_numpy_brazil(paths[0])
        # noisy_image2, clean_image2 = csv_to_numpy_brazil(paths[1])
    
        if PLOT_IMAGES:
            plot_grid(noisy_image1, SAVE_DIR, "noisy_image1")
            plot_grid(clean_image1, SAVE_DIR, "original_image1")
            print(noisy_image1.shape, clean_image1.shape)
            plot_grid(noisy_image2, SAVE_DIR, "noisy_image2")
            plot_grid(clean_image2, SAVE_DIR, "original_image2")
            print(noisy_image2.shape, clean_image2.shape)


        # if reconstruct_grid == 'train':
        #     reconstruct_dataset = BrazilDataset(data=(data1, data2), split='reconstruct', patch_dim=IMAGE_SIZE, transform=val_transform)
        # elif reconstruct_grid == 'val':
        #     reconstruct_dataset = BrazilDataset(data=(data1, data2), split='reconstruct', patch_dim=IMAGE_SIZE, transform=val_transform)
        # elif reconstruct_grid == 'test':
        #     reconstruct_dataset = BrazilDataset(data=(data1, data2), split='reconstruct', patch_dim=IMAGE_SIZE, transform=val_transform)
        # else:
        #     reconstruct_dataset = BrazilDataset(data=(data1, data2), split='reconstruct', patch_dim=IMAGE_SIZE, transform=val_transform)



        seed = 42

        train_dataset = BrazilDataset_v2(datasets=train_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = IMAGE_SIZE//32, transform=train_transform, seed=seed, split = 'train')
        val_dataset = BrazilDataset_v2(datasets=val_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = IMAGE_SIZE//32, transform=val_transform, seed=seed, split = 'val')
        test_dataset = BrazilDataset_v2(datasets=test_data, patch_dim=IMAGE_SIZE, fill_value=fill, stride = IMAGE_SIZE//test_overlap, transform=val_transform, seed=seed, split = 'test')
        
        if DDP==False:
            sampler_train = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
            sampler_val = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank)


            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler = sampler_train, shuffle=True, num_workers=NUM_WORKERS)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler = sampler_val, shuffle=False, num_workers=NUM_WORKERS)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler = sampler_val, shuffle=False, num_workers=NUM_WORKERS)

            return train_loader, val_loader, test_loader, test_loader,0, 0 # extremes, pads


















    print(DATASET, len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last = drop_last)

    return train_loader, val_loader, test_loader, test_loader, train_dataset, val_dataset, test_dataset





















import numpy as np
import csv
import csv
import numpy as np
import csv
import numpy as np

def efficient_csv_to_numpy(filename):
    x_values = set()
    y_values = set()
    z_values = []
    clean_values = []
    noisy_values = []

    # First pass: collect unique X, Y values and Z, clean, noisy values
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header row
        x_index = header.index('X')
        y_index = header.index('Y')
        noisy_index = header.index('MAGCOR_IGRF')
        clean_index = header.index('MAGIGRF')

        for line in reader:
            x = float(line[x_index])
            y = float(line[y_index])
            noisy = float(line[noisy_index])
            clean = float(line[clean_index])

            x_values.add(x)
            y_values.add(y)
            clean_values.append(clean)
            noisy_values.append(noisy)

    # Determine the shape of the grid
    num_x = len(x_values)
    num_y = len(y_values)

    # Check if the dimensions are consistent
    if len(clean_values) != num_x * num_y:
        raise ValueError("Mismatch between Z values count and X, Y dimensions.")

    # Convert lists of values to NumPy arrays and reshape
    clean_array = np.array(clean_values).reshape((num_y, num_x))
    noisy_array = np.array(noisy_values).reshape((num_y, num_x))

    return clean_array, noisy_array


def plot_metrics(final_results, save_dir, name=None):
    # Extract PSNR, SSIM, and L1 values
    noisy_psnr = [res[0][0] for res in final_results]
    denoised_psnr = [res[0][1] for res in final_results]
    
    noisy_ssim = [res[1][0] for res in final_results]
    denoised_ssim = [res[1][1] for res in final_results]
    
    noisy_l1 = [res[2][0] for res in final_results]
    denoised_l1 = [res[2][1] for res in final_results]
    
    epochs = range(0, len(final_results))
    
    # Plot PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, noisy_psnr, marker='o', linestyle='-', label='Noisy PSNR', color='b')
    plt.plot(epochs, denoised_psnr, marker='s', linestyle='--', label='Denoised PSNR', color='r')
    plt.title('PSNR Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(save_dir + '/' + (name + '_PSNR.png' if name else 'PSNR.png'))
    plt.close()
    
    # Plot SSIM
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, noisy_ssim, marker='o', linestyle='-', label='Noisy SSIM', color='b')
    plt.plot(epochs, denoised_ssim, marker='s', linestyle='--', label='Denoised SSIM', color='r')
    plt.title('SSIM Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(save_dir + '/' + (name + '_SSIM.png' if name else 'SSIM.png'))
    plt.close()
    
    # Plot L1 Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, noisy_l1, marker='o', linestyle='-', label='Noisy L1', color='b')
    plt.plot(epochs, denoised_l1, marker='s', linestyle='--', label='Denoised L1', color='r')
    plt.title('L1 Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L1 Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(save_dir + '/' + (name + '_L1.png' if name else 'L1.png'))
    plt.close()



def plot_results(results, save_dir, name = None):

    plt.figure(figsize=(10, 6))
    plt.plot(results['train_loss'], marker='o', linestyle='-', label='Train Loss', color='b')
    plt.plot(results['val_loss'], marker='s', linestyle='--', label='Validation Loss', color='r')
    plt.plot(results['test_loss'], marker='d', linestyle=':', label='Test Loss', color='g')

    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.title('Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.plot(results['train_acc'], label='Train accuracy')
    # plt.plot(results['val_acc'], label='Validation accuracy')
    # plt.plot(results['train_f1'], label='Train F1')
    # plt.plot(results['train_recall'], label='Train Recall')
    
    # plt.plot(results['val_f1'], label='Val F1')
    # plt.plot(results['val_recall'], label='Val Recall')
    # plt.plot(results['train_kappa'], label='Train Kappa')
    # plt.plot(results['val_kappa'], label='Val Kappa')
    # plt.legend()
    if name:
        plt.savefig(save_dir + name)
    else:
        plt.savefig(save_dir + '/LossAccuracy.png')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            print(self.counter)
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def csv_to_numpy(csv_path):
    df = pd.read_csv(csv_path)
    x_unique = np.sort(df['X'].unique())
    y_unique = np.sort(df['Y'].unique())
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    z_grid = np.full_like(x_grid, fill_value=np.nan, dtype=float)
    # df['Z'] = df['Z'] - df['Z'].mean()
    print(df['Z'])
    for index, row in df.iterrows():
        x_idx = np.where(x_unique == row['X'])[0][0]
        y_idx = np.where(y_unique == row['Y'])[0][0]
        z_grid[y_idx, x_idx] = row['Z']
    return z_grid



def csv_to_numpy_brazil(csv_path):
    df = pd.read_csv(csv_path)
    x_unique = np.sort(df['X'].unique())
    y_unique = np.sort(df['Y'].unique())
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    noisy_grid = np.full_like(x_grid, fill_value=np.nan, dtype=float)
    clean_grid = np.full_like(x_grid, fill_value=np.nan, dtype=float)
    for index, row in df.iterrows():
        x_idx = np.where(x_unique == row['X'])[0][0]
        y_idx = np.where(y_unique == row['Y'])[0][0]
        noisy_grid[y_idx, x_idx] = row['MAGCORIGRF']
        clean_grid[y_idx, x_idx] = row['MAGIGRF']
    return noisy_grid, clean_grid


def split_grid(grid, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    total_rows = grid.shape[0] #251
    train_end = int(train_ratio * total_rows) #0.6*251 = 150
    test_end = int((train_ratio + test_ratio) * total_rows) #int(0.8*251) = 200

    train_grid = grid[0:train_end, :] #grid[:, 251 - 150:] = [:, 101:]
    val_grid = grid[train_end+1:, 0:grid.shape[1]//2] #grid[0:400, 0:(251-150)]
    test_grid = grid[train_end+1:, grid.shape[1]//2+1:] #grid[0:400, 0:(251-150)]

    return train_grid, test_grid, val_grid

# def split_grid(grid, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
#     total_rows = grid.shape[0]
#     train_end = int(train_ratio * total_rows)
#     test_end = int((train_ratio + test_ratio) * total_rows)
#     train_grid = grid[:, grid.shape[0] - train_end:]
#     val_grid = grid[0:grid.shape[1]//2, 0:grid.shape[0] - train_end]
#     test_grid = grid[grid.shape[1]//2:, 0:grid.shape[0] - train_end]
#     return train_grid, test_grid, val_grid

def save_subplot(noisy, outputs, original, save_path, num_images=8):
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    for i in range(num_images):
        # Convert tensors to numpy arrays and handle channel dimensions
        noisy_img = noisy[i].cpu().numpy()
        output_img = outputs[i].cpu().numpy()
        original_img = original[i].cpu().numpy()
        
        # Handle grayscale images
        if noisy_img.shape[0] == 1:
            noisy_img = noisy_img.squeeze(0)
            output_img = output_img.squeeze(0)
            original_img = original_img.squeeze(0)
            axs[i, 0].imshow(noisy_img, cmap='gray')
            axs[i, 1].imshow(output_img, cmap='gray')
            axs[i, 2].imshow(original_img, cmap='gray')
        # Handle RGB images
        elif noisy_img.shape[0] == 3:
            noisy_img = np.transpose(noisy_img, (1, 2, 0))  # Change shape from (3, H, W) to (H, W, 3)
            output_img = np.transpose(output_img, (1, 2, 0))
            original_img = np.transpose(original_img, (1, 2, 0))
            axs[i, 0].imshow(noisy_img)
            axs[i, 1].imshow(output_img)
            axs[i, 2].imshow(original_img)
        else:
            noisy_img = noisy_img[0]
            output_img = output_img.squeeze(0)
            original_img = original_img.squeeze(0)
            axs[i, 0].imshow(noisy_img, cmap='gray')
            axs[i, 1].imshow(output_img, cmap='gray')
            axs[i, 2].imshow(original_img, cmap='gray')  
            # raise ValueError("Unsupported image shape")

        axs[i, 0].set_title('Noisy')
        axs[i, 0].axis('off')
        axs[i, 1].set_title('Denoised')
        axs[i, 1].axis('off')
        axs[i, 2].set_title('Original')
        axs[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_grid(grid, save_dir, fig_name):
    unique_x = np.arange(grid.shape[0])  # Number of columns
    unique_y = np.arange(grid.shape[1])  # Number of rows
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(unique_x, unique_y, grid.T, cmap='gray')
    plt.colorbar(label='Z values')
    plt.xlabel('Y_numpy')
    plt.ylabel('X_numpy')
    plt.title('2D Grid of Z values')
    plt.savefig(f'{save_dir}/{fig_name}.png')
    plt.close()


def save_images_with_subplots(original_images, noisy_actual_images, noisy_images, save_path, epoch, start_idx):
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, f'{epoch}_reconstructed'))
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(f'Epoch {epoch} - Image {start_idx + idx}')
    axes[0].imshow(original_images, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(noisy_actual_images, cmap='gray')
    axes[1].set_title('Noisy Image')
    axes[1].axis('off')
    axes[2].imshow(noisy_images, cmap='gray')
    axes[2].set_title('Denoised Image')
    axes[2].axis('off')
    plt.savefig(os.path.join(os.path.join(save_path, f'{epoch}_reconstructed'), f"{epoch}_image_{start_idx}.png"))
    plt.close()

def normalize(tensor):
    mask = (tensor >= 0) & (tensor <= 1)
    filtered_tensor = tensor[mask]
    if filtered_tensor.nelement()==0:
        return tensor,torch.tensor(0.),torch.tensor(0.)
    # Step 2: Find the min and max within the filtered range
    min_val = filtered_tensor.min()
    max_val = filtered_tensor.max()

    normalized_tensor = torch.where(mask, (tensor - min_val) / ((max_val - min_val) + 1e-10), tensor)
    return normalized_tensor, min_val, max_val


def divide_and_resize_v2(grid):
    patches_noisy = []
    patches_original = []
    patch_width, patch_height = 256,256
    grid = grid.permute(0,3,1,2)
    minis = []
    maxis = []
    for i in range(grid.shape[2]//256):
        i = i* 256
        for j in range(grid.shape[3]//256):
            j = j * 256
            patch_noisy = grid[:,0,:,:][:,i:i + patch_height, j:j + patch_width]
            patch_original = grid[:,1,:,:][:,i:i + patch_height, j:j + patch_width]
            patch_noisy_v2, mini_noisy, maxi_noisy = normalize(patch_noisy)
            patch_original_v2, mini_ori, maxi_ori = normalize(patch_original)
            patch_noisy_v2 = patch_noisy
            patches_noisy.append(patch_noisy_v2.float())
            patches_original.append(patch_original.float())
            minis.append(mini_noisy)
            maxis.append(maxi_noisy)
    return torch.stack(patches_noisy), torch.stack(patches_original), torch.stack(minis), torch.stack(maxis), grid.shape[3], grid.shape[2]


def reconstruct(patches_noisy, patches_original, patches_actual_noisy, width, height, minis,maxis):#, pads):
    patch_width, patch_height = 256, 256
    reconstructed_noisy = np.zeros((height, width), dtype=np.float16)
    reconstructed_original = np.zeros((height, width), dtype=np.float16)
    reconstructed_noisy_img = np.zeros((height, width), dtype=np.float16)
    patch_index = 0
    for j in range(height//256):
        for i in range(width//256):
            left = i * patch_width
            upper = j * patch_height
            patch_noisy = patches_noisy[patch_index]
            patch_original = patches_original[patch_index]
            patch_actual_noisy = patches_actual_noisy[patch_index]
            # print('before', patch_noisy,patch_noisy.max(), patch_noisy.min(), maxis[patch_index], minis[patch_index])
            # mask = (patch_noisy == 0.00)  # Create a mask for elements that are not equal to 0
            mask = torch.abs(patch_noisy) < 0.09
            patch_noisy[~mask] = patch_noisy[~mask] * (maxis[patch_index] - minis[patch_index] + 1e-10) + minis[patch_index]
            patch_noisy[mask] = 0
            # patch_noisy = patch_noisy * (maxis[patch_index] - minis[patch_index] + 1e-10) + minis[patch_index]
            if not isinstance(patch_noisy, np.ndarray):
                patch_noisy = np.array(patch_noisy.cpu())
            if not isinstance(patch_original, np.ndarray):
                patch_original = np.array(patch_original.cpu())
                patch_actual_noisy = np.array(patch_actual_noisy.cpu())
            reconstructed_noisy[upper:upper + patch_height, left:left + patch_width] = patch_noisy[0]
            reconstructed_original[upper:upper + patch_height, left:left + patch_width] = patch_original[0]
            reconstructed_noisy_img[upper:upper + patch_height, left:left + patch_width] = patch_actual_noisy[0]



            patch_index += 1
    # reconstructed_noisy = reconstructed_noisy[pads[0]:reconstructed_noisy.shape[0] - pads[1], pads[3]: reconstructed_noisy.shape[1] - pads[2]]
    # reconstructed_original = reconstructed_original[pads[0]:reconstructed_original.shape[1] - pads[3], pads[3]: reconstructed_original.shape[1] - pads[2]]
    # reconstructed_noisy_img = reconstructed_noisy_img[pads[0]:reconstructed_noisy_img.shape[1] - pads[3], pads[3]: reconstructed_noisy_img.shape[1] - pads[2]]

    return reconstructed_noisy, reconstructed_original, reconstructed_noisy_img




def preprocess_grid(grid,IMAGE_SIZE, fill = None):
    mini, maxi = np.nanmin(grid[:,:,0]), np.nanmax(grid[:,:,0])
        
    new_width = math.ceil(grid.shape[0]/IMAGE_SIZE) * IMAGE_SIZE
    new_height = math.ceil(grid.shape[1]/IMAGE_SIZE) * IMAGE_SIZE
    
    # grid[:,:,0] = (grid[:,:,0] - np.nanmin(grid[:,:,0]))/(np.nanmax(grid[:,:,0]) - np.nanmin(grid[:,:,0]))
    # grid[:,:,1] = (grid[:,:,1] - np.nanmin(grid[:,:,1]))/(np.nanmax(grid[:,:,1]) - np.nanmin(grid[:,:,1]))
    grid = np.nan_to_num(grid, nan=fill)

    pad_left = (new_width - grid.shape[0]) // 2
    pad_right = new_width - grid.shape[0] - pad_left
    pad_top = (new_height - grid.shape[1]) // 2
    pad_bottom = new_height - grid.shape[1] - pad_top
    grid_v2 = np.pad(grid, pad_width=((pad_left, pad_right), (pad_top, pad_bottom), (0,0) ), mode='constant', constant_values=fill)

    return grid_v2, [mini, maxi], [pad_left, pad_right, pad_top, pad_bottom]

















class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if not self.normalize:
            print('normalize skipped.')

        if initial_weights is not None and type(initial_weights) == tuple:
            print('tuple.')
            w, b = initial_weights
            self.classification_head.weight = torch.nn.Parameter(w.clone())
            self.classification_head.bias = torch.nn.Parameter(b.clone())
        else:
            if initial_weights is None:
                initial_weights = torch.zeros_like(self.classification_head.weight)
                torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
            self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
            # Note: modified. Initial bug in forgetting to zero bias.
            self.classification_head.bias = torch.nn.Parameter(torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model.encode_image(images).float()
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        return logits
    


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)



def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(tqdm((train_loader))):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.to('cuda')
        target = target.to('cuda')

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    model = model.to('cuda')
    model.eval()
    # print(model)
    # print('hooooodfshdkfdfjhsdkjg')
    with torch.no_grad():
        for input, target in test_loader:
            # print('hooooo')
            input = input.to('cuda')
            target = target.to('cuda')

            output = model(input, **kwargs)
            # print(output)
            nll = criterion(output, target)
            loss = nll.clone()
            if regularizer is not None:
                loss += regularizer(model)

            nll_sum += nll.item() * input.size(0)
            loss_sum += loss.item() * input.size(0)
            pred = torch.nn.functional.softmax(output).data.argmax(1, keepdim=True)
            # print(pred,target)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            # print(correct)
    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


import warnings
import pdb
def fft_transform(array):
    np.seterr(all='warn')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fourier_transform = np.fft.fft2(array)
            center_shift = np.fft.fftshift(fourier_transform)

            fourier_noisy = np.abs(center_shift) #20 * np.log(np.abs(center_shift))
            min_val = fourier_noisy.min()
            max_val = fourier_noisy.max()
            fourier_noisy = (fourier_noisy - min_val) / (max_val - min_val + 1e-5)

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

        # Check for warnings after the computations
        for warning in w:
            if issubclass(warning.category, RuntimeWarning):
                print(f"RuntimeWarning: {warning.message} at {warning.filename}:{warning.lineno}")
                pdb.set_trace()  # Set breakpoint here
    
    return torch.from_numpy(fourier_noisy)



def postprocess(noisy_img, save_dir, paths, csv=''):
    if csv!='':
        noisy_image_T = noisy_img[0]
        rows, cols = noisy_image_T.shape
        df_ori = pd.read_csv(csv)#/home/santosh/Projects/geo_physics/meixia_noise_removal/brazil_v2_data/csvs/test1_1111.csv
        df = pd.DataFrame({'X': np.repeat(np.arange(rows), cols), 'Y': np.tile(np.arange(cols), rows),'MAGIGRF': noisy_image_T.flatten()})
        df_cleaned = df.dropna(subset=['MAGIGRF'])
        df_cleaned = df_cleaned.reset_index(drop=True)
        df_cleaned[['X', 'Y']] = df_ori[['X', 'Y']]
        df_cleaned.to_csv(os.path.join(save_dir, paths), index=False)

# def   merge_patches_with_median(patches, patch_size, image_size, overlap, mode = 'median'):
#     """
#     Merges overlapping patches into a single image using median blending.

#     Args:
#     - patches (list of np.ndarray): List of patches to be merged.
#     - patch_size (tuple): Size of each patch (height, width).
#     - image_size (tuple): Size of the final image (height, width).
#     - overlap (int): Number of pixels of overlap between patches.

#     Returns:
#     - np.ndarray: Merged image.
#     """
#     # Initialize a list to store values for median calculation
#     merged_image_values = np.empty((image_size[1], image_size[0], 0), dtype=np.float32)

#     patch_height, patch_width = patch_size, patch_size
#     image_height, image_width = image_size[1], image_size[0]
#     step_size = patch_height - overlap  # Calculate the step size

#     # Initialize an empty canvas to store all pixel contributions
#     contribution_list = [[[] for _ in range(image_width)] for _ in range(image_height)]

#     patch_idx = 0
#     # Iterate through the image placing patches
#     for y in range(0, image_height - patch_height + 1, step_size):
#         for x in range(0, image_width - patch_width + 1, step_size):
#             # Get the current patch
#             patch = np.array(patches[patch_idx])[0]
#             patch_idx += 1
            
#             # Add patch to the corresponding positions
#             for i in range(patch_height):
#                 for j in range(patch_width):
#                     contribution_list[y + i][x + j].append(patch[i, j])
#             # print('contriiiiiiiii',contribution_list[y][x])
#     # Calculate the median of all contributions at each pixel
#     merged_image = np.zeros((image_size[1],image_size[0]), dtype=np.float32)
#     for i in range(image_height):
#         for j in range(image_width):
#             if contribution_list[i][j]:
#                 # Filter contributions to be within 2 standard deviations from the mean
#                 if mode == 'mean':
#                     data = np.array(contribution_list[i][j])
#                     mean_value = np.mean(data)
#                     std_value = np.std(data)
#                     filtered_data = data[(data >= mean_value - 2 * std_value) & (data <= mean_value + 2 * std_value)]
                    
#                     if len(filtered_data) > 0:
#                         merged_image[i, j] = np.mean(filtered_data)
#                     else:
#                         merged_image[i, j] = mean_value  # Fallback to mean if no data within 2 std
#                 else:
#                     merged_image[i, j] = np.median(contribution_list[i][j])


#     return merged_image

def merge_patches_with_median(patches, patch_size, image_size, overlap, mode='median'):
    """
    Merges overlapping patches into a single image using median or mean blending.

    Args:
    - patches (list of np.ndarray): List of patches to be merged.
    - patch_size (tuple): Size of each patch (height, width).
    - image_size (tuple): Size of the final image (height, width).
    - overlap (int): Number of pixels of overlap between patches.
    - mode (str): 'mean' or 'median' for the blending method.

    Returns:
    - np.ndarray: Merged image.
    """
    patch_height, patch_width = patch_size, patch_size
    image_height, image_width = image_size[1], image_size[0]
    step_size = patch_height - overlap  # Calculate the step size

    # Initialize arrays to store contributions and counts
    contribution_values = np.zeros((image_height, image_width), dtype=np.float32)
    contribution_count = np.zeros((image_height, image_width), dtype=int)

    patch_idx = 0
    # Iterate through the image placing patches
    for y in range(0, image_height - patch_height + 1, step_size):
        for x in range(0, image_width - patch_width + 1, step_size):
            patch = np.array(patches[patch_idx])[0]
            patch_idx += 1

            # Add patch values to the contribution arrays directly
            contribution_values[y:y+patch_height, x:x+patch_width] += patch
            contribution_count[y:y+patch_height, x:x+patch_width] += 1

    # Now compute the final merged image
    merged_image = np.zeros((image_height, image_width), dtype=np.float32)

    if mode == 'mean':
        # Efficient mean computation by dividing sum by count
        # Avoid division by zero by using np.where
        merged_image = np.divide(contribution_values, contribution_count, where=contribution_count != 0)


    elif mode == 'median':
        # Create a 3D list to store pixel values efficiently
        max_patches = (image_height // step_size) * (image_width // step_size)
        pixel_stack = np.full((image_height, image_width, max_patches), np.nan, dtype=np.float32)

        patch_idx = 0
        for y in range(0, image_height - patch_height + 1, step_size):
            for x in range(0, image_width - patch_width + 1, step_size):
                patch = patches[patch_idx]
                pixel_stack[y:y + patch_height, x:x + patch_width, patch_idx] = patch
                patch_idx += 1
        # Compute median along the last axis, ignoring NaN values
        merged_image = np.nanmedian(pixel_stack, axis=-1)

    return merged_image