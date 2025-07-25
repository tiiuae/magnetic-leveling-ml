import torch

from tqdm import tqdm
import os
import json
from utils.train_utils import get_weights
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from utils.utils import fft_transform, save_subplot, reconstruct, divide_and_resize_v2, merge_patches_with_median
import cv2
import matplotlib.image
import random
import numpy as np
import ignite.metrics
import numpy.ma as ma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import torch
import numpy.ma as ma
import matplotlib.image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sewar.full_ref import msssim  # Install with: pip install sewar

def evaluate_denoising(original, noisy, denoised):
    """
    Compute various denoising quality metrics given original, noisy, and denoised images.

    Args:
        original (np.ndarray): The original clean image.
        noisy (np.ndarray): The noisy input image.
        denoised (np.ndarray): The denoised output image.

    Returns:
        list: A list containing the original, noisy, and denoised images,
              along with their respective metrics.
    """
    
    # Ensure images are in float format
    original = original.astype(np.float32)
    noisy = noisy.astype(np.float32)
    denoised = denoised.astype(np.float32)

    # Compute PSNR
    noisy_psnr = psnr(original, noisy, data_range=noisy.max() - noisy.min())
    denoised_psnr = psnr(original, denoised, data_range=denoised.max() - denoised.min())

    # Compute SSIM
    noisy_ssim = ssim(original, noisy, data_range=noisy.max() - noisy.min())
    denoised_ssim = ssim(original, denoised, data_range=denoised.max() - denoised.min())

    # Compute MAE (Mean Absolute Error)
    noisy_l1 = np.mean(np.abs(noisy - original))
    denoised_l1 = np.mean(np.abs(denoised - original))

    # Compute MSE (Mean Squared Error)
    noisy_mse = np.mean((noisy - original) ** 2)
    denoised_mse = np.mean((denoised - original) ** 2)

    # Compute RMSE (Root Mean Squared Error)
    noisy_rmse = np.sqrt(noisy_mse)
    denoised_rmse = np.sqrt(denoised_mse)

    # Compute Residual Energy Ratio (RER) (Lower is better)
    residual_original = noisy - original  # True noise
    residual_denoised = noisy - denoised  # Leftover noise after denoising
    rer = np.sum(residual_denoised ** 2) / np.sum(residual_original ** 2)

    # Compute SNR (Signal-to-Noise Ratio) for Noisy and Denoised images
    var_original = np.var(original)
    var_residual_noisy = np.var(residual_original)
    var_residual_denoised = np.var(residual_denoised)
    
    snr_noisy = 10 * np.log10(var_original / var_residual_noisy)
    snr_denoised = 10 * np.log10(var_original / var_residual_denoised)
    delta_snr = snr_denoised - snr_noisy  # Improvement in SNR

    # Compute FSIM (Feature Similarity Index)
    noisy_mssim = msssim(original, noisy, MAX=noisy.max() - noisy.min())
    denoised_mssim = msssim(original, denoised, MAX=denoised.max() - denoised.min())



    # Print all metrics with labels
    print("Metrics for Image Denoising:")
    print("Noisy Image vs. Denoised Image vs. Original Image:")
    print(f"PSNR (Noisy vs Original): {noisy_psnr:.4f}, PSNR (Denoised vs Original): {denoised_psnr:.4f}")
    print(f"SSIM (Noisy vs Original): {noisy_ssim:.4f}, SSIM (Denoised vs Original): {denoised_ssim:.4f}")
    print(f"L1 Loss (Noisy vs Original): {noisy_l1:.4f}, L1 Loss (Denoised vs Original): {denoised_l1:.4f}")
    print(f"MSE (Noisy vs Original): {noisy_mse:.4f}, MSE (Denoised vs Original): {denoised_mse:.4f}")
    print(f"RMSE (Noisy vs Original): {noisy_rmse:.4f}, RMSE (Denoised vs Original): {denoised_rmse:.4f}")
    print(f"Relative Error Ratio (RER): {rer:.4f}")
    print(f"SNR (Noisy): {snr_noisy:.4f}, SNR (Denoised): {snr_denoised:.4f}")
    print(f"Delta SNR (Denoised vs Noisy): {delta_snr:.4f}")
    print(f"MSSIM (Noisy vs Original): {noisy_mssim:.4f}, MSSIM (Denoised vs Original): {denoised_mssim:.4f}")


    # Return the results in the specified format
    return [denoised, original, noisy, 
            [
                [noisy_psnr, denoised_psnr], 
                [noisy_ssim, denoised_ssim], 
                [noisy_l1, denoised_l1], 
                [noisy_mse, denoised_mse], 
                [noisy_rmse, denoised_rmse], 
                [rer, rer],
                [snr_noisy, snr_denoised], 
                [delta_snr, delta_snr], 
                [noisy_mssim, denoised_mssim], 
            ]]

def PSNR(preds, targets):
     maxi = max(preds.max(), targets.max())
     mini = min(preds.min(), targets.min())
     psnr = ignite.metrics.PSNR(data_range =maxi - mini)
     psnr.update((preds, targets))
     return psnr.compute()
def SSIM(preds, targets):
     maxi = max(preds.max(), targets.max())
     mini = min(preds.min(), targets.min())
     psnr = ignite.metrics.SSIM(data_range = maxi - mini)
     psnr.update((preds, targets))
     return psnr.compute()

def L1(preds, targets):
     mae = ignite.metrics.MeanAbsoluteError()
     mae.update((preds, targets))
     return mae.compute()



def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

START_seed()


def train_step(
        model: torch.nn.Module,
        train_loader,
        train_loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lr_scheduler = None,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the epoch.
    """

    model.train()
    train_loss = 0.0
    
    num_iters = len(train_loader)
    for iter,data in enumerate(tqdm(train_loader)):
        noisy_patches, original_patches,_,_ = data
        noisy_patches = noisy_patches.to(device).float()
        original_patches = original_patches.to(device).float()
        outputs = model(noisy_patches).to(device).float()
        loss = train_loss_fn(outputs, original_patches)
        # loss+= train_loss_fn(outputs, noisy_patches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * noisy_patches.size(0)
    train_loss /= len(train_loader.dataset)

    return train_loss


def val_step(
        model: torch.nn.Module,
        val_loader,
        train_loader,
        val_loss_fn: torch.nn.Module,
        device: torch.device,
        epoch,
        val_per_epoch,
        save_dir
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the validation set.
    """

    model.eval()
    val_loss = 0.0


    with torch.no_grad():
        for iter,data in enumerate(tqdm(val_loader)):
            noisy_patches, original_patches,_,_ = data
            noisy_patches = noisy_patches.to(device).float()
            original_patches = original_patches.to(device).float()
            outputs = model(noisy_patches).to(device).float()
            loss = val_loss_fn(outputs, original_patches)
            # loss+= val_loss_fn(outputs, noisy_patches)
            val_loss += loss.item() * noisy_patches.size(0)
        val_loss /= len(val_loader.dataset)


    if epoch%val_per_epoch==0:
        num_images = 8
        indices = torch.randperm(noisy_patches.size(0))
        noisy_patches_subset = noisy_patches[indices]
        outputs_subset = outputs[indices]
        original_patches_subset = original_patches[indices]
        save_path = os.path.join(save_dir, f'epoch_val_{epoch}_denoise.png')
        save_subplot(noisy_patches_subset, outputs_subset, original_patches_subset, save_path, num_images)

    return val_loss




def test_step(
        model: torch.nn.Module,
        val_loader,
        train_loader,
        val_loss_fn: torch.nn.Module,
        device: torch.device,
        epoch,
        val_per_epoch,
        save_dir
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the validation set.
    """

    model.eval()
    val_loss = 0.0


    with torch.no_grad():
        for iter,data in enumerate(tqdm(val_loader)):
            noisy_patches, original_patches,_,_ = data
            noisy_patches = noisy_patches.to(device).float()
            original_patches = original_patches.to(device).float()
            outputs = model(noisy_patches).to(device)
            loss = val_loss_fn(outputs, original_patches)
            val_loss += loss.item() * noisy_patches.size(0)
        val_loss /= len(val_loader.dataset) * 8

        num_images = 8
        indices = torch.randperm(noisy_patches.size(0))
        noisy_patches_subset = noisy_patches[indices]
        outputs_subset = outputs[indices]
        original_patches_subset = original_patches[indices]
        save_path = os.path.join(save_dir, f'epoch_test_{epoch}_denoise.png')
        save_subplot(noisy_patches_subset, outputs_subset, original_patches_subset, save_path, num_images)

    return val_loss



def rec_step(model, test_dataset, reconstruct_loader,device, save_dir, IMAGE_SIZE, pads=None, test_overlap = None):
    list_of_noisy_patches = []
    list_of_original_patches = []
    list_of_denoised_patches = []
    minis_list = []
    maxis_list = []
    final_dir = save_dir.get('final_dir')
    file_name = os.path.basename(str(save_dir.get('path')))[:-4]
    print(final_dir, file_name)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(reconstruct_loader), desc=f"Test denoising", total=len(reconstruct_loader)):
            noisy_patches, original_patches, [mini_noisy, mini_clean, maxi_noisy, maxi_clean], [width, height] = data
            noisy_patches = noisy_patches.float().to(device)
            original_patches = original_patches.float().to(device)
            outputs = model(noisy_patches).to(device)
            
            outputs = (outputs.float().to(device) * (maxi_noisy.float().to(device) - mini_noisy.float().to(device))) + mini_noisy.float().to(device)
            noisy_patches = torch.where(noisy_patches == 0, 0, (noisy_patches.float().to(device) * (maxi_noisy.float().to(device) - mini_noisy.float().to(device)) + mini_noisy.float().to(device))) 
            original_patches = torch.where(original_patches == 0, 0, (original_patches.float().to(device) * (maxi_clean.float().to(device) - mini_clean.float().to(device)) + mini_clean.float().to(device))) 

            list_of_denoised_patches.append(outputs.cpu())
            list_of_noisy_patches.append(noisy_patches.cpu())
            list_of_original_patches.append(original_patches.cpu())
            minis_list.append(mini_clean.float().to(device))
            maxis_list.append(maxi_clean.float().to(device))

        denoised_stack = torch.cat(list_of_denoised_patches, dim=0)
        noisy_stack = torch.cat(list_of_noisy_patches, dim=0)
        original_stack = torch.cat(list_of_original_patches, dim=0)
        # save_path = os.path.join(save_dir, f'epoch_test_test_denoise.png')
        denoised_image = merge_patches_with_median(denoised_stack, IMAGE_SIZE, [width, height], IMAGE_SIZE - test_overlap, mode='median')
        # original_image = merge_patches_with_median(original_stack, IMAGE_SIZE, [width, height], IMAGE_SIZE - test_overlap, mode='median')
        # noisy_image = merge_patches_with_median(noisy_stack, IMAGE_SIZE, [width, height], IMAGE_SIZE - test_overlap, mode='median')
        # original_array = np.where(original_image == 0, np.nan, original_image)
        # noisy_array = np.where(noisy_image == 0, np.nan, noisy_image)
        denoised_array = np.where(np.isnan(test_dataset.padded_data[:,:,1]), np.nan, denoised_image)

        if test_dataset.padding_info[0][1]!=0:
            if test_dataset.padding_info[0][3]!=0:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                # original_array = original_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                # noisy_array = noisy_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
            else:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:]
                # original_array = original_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:]
                # noisy_array = noisy_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:]
        else:
            if test_dataset.padding_info[0][3]!=0:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                # original_array = original_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                # noisy_array = noisy_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
            else:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:]
                # original_array = original_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:]
                # noisy_array = noisy_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:]

        # masked_array = ma.masked_invalid(noisy_array)
        # matplotlib.image.imsave(f'{final_dir}/noisy_{file_name}.png', masked_array)
        # masked_array = ma.masked_invalid(original_array)
        # matplotlib.image.imsave(f'{final_dir}/original_{file_name}.png', masked_array)



        # masked_array = ma.masked_invalid(denoised_array)
        # matplotlib.image.imsave(f'{final_dir}/denoised_{file_name}.png', masked_array)
        # original_array = np.load(save_dir.get('path'))[:,:,1]
        # noisy_array = np.load(save_dir.get('path'))[:,:,0]
        # mask = ~np.isnan(denoised_array) & ~np.isnan(original_array) & ~np.isnan(noisy_array)

        # denoised_tensor = torch.tensor(denoised_array, dtype=torch.float32)
        # original_tensor = torch.tensor(original_array, dtype=torch.float32)
        # noisy_tensor = torch.tensor(noisy_array, dtype=torch.float32)

        # denoised_tensor[~mask] = 0
        # original_tensor[~mask] = 0
        # noisy_tensor[~mask] = 0
        # H, W = denoised_array.shape  # Replace with your actual image size
        # denoised_tensor = denoised_tensor.view(1, 1, H, W)
        # original_tensor = original_tensor.view(1, 1, H, W)
        # noisy_tensor = noisy_tensor.view(1, 1, H, W)

        # # noisy_tensor = (noisy_tensor - noisy_tensor.min())/ (noisy_tensor.max() - noisy_tensor.min())
        # # original_tensor = (original_tensor - original_tensor.min())/ (original_tensor.max() - original_tensor.min())
        # # denoised_tensor = (denoised_tensor - denoised_tensor.min())/ (denoised_tensor.max() - denoised_tensor.min())


        # noisy_psnr, denoised_psnr = PSNR(noisy_tensor, original_tensor), PSNR(denoised_tensor, original_tensor)
        # noisy_ssim, denoised_ssim = SSIM(noisy_tensor, original_tensor), SSIM(denoised_tensor, original_tensor)
        # noisy_l1, denoised_l1 = L1(noisy_tensor, original_tensor)/ (original_tensor.shape[2] * original_tensor.shape[3]), L1(denoised_tensor, original_tensor)/ (original_tensor.shape[2] * original_tensor.shape[3])

        # print(f"Noisy PSNR: {noisy_psnr}, Denoised PSNR: {denoised_psnr}")
        # print(f"Noisy SSIM: {noisy_ssim}, Denoised SSIM: {denoised_ssim}")
        # print(f"Noisy L1: {noisy_l1}, Denoised L1: {denoised_l1}")


        # # Mask invalid values
        # masked_array = ma.masked_invalid(denoised_array)
        # matplotlib.image.imsave(f'{final_dir}/denoised_{file_name}.png', masked_array)

        # # Load arrays
        # original_array = np.load(save_dir.get('path'))[:, :, 1]
        # noisy_array = np.load(save_dir.get('path'))[:, :, 0]

        # # Create mask
        # mask = ~np.isnan(denoised_array) & ~np.isnan(original_array) & ~np.isnan(noisy_array)

        # # Convert to tensors
        # # denoised_tensor = torch.tensor(denoised_array, dtype=torch.float32)
        # # original_tensor = torch.tensor(original_array, dtype=torch.float32)
        # # noisy_tensor = torch.tensor(noisy_array, dtype=torch.float32)

        # # Apply mask
        # denoised_array[~mask] = 0
        # original_array[~mask] = 0
        # noisy_array[~mask] = 0

        # # # Reshape tensors
        # # H, W = denoised_array.shape
        # # denoised_tensor = denoised_tensor.view(1, 1, H, W)
        # # original_tensor = original_tensor.view(1, 1, H, W)
        # # noisy_tensor = noisy_tensor.view(1, 1, H, W)
        # # denoised_array = denoised_array.astype(np.float32)
        # # original_array = original_array.astype(np.float32)
        # # noisy_array = noisy_array.astype(np.float32)
        # # # Calculate PSNR, SSIM, L1

        # # # maxi = max(denoised_array.max(), original_array.max(), noisy_array.max())
        # # # mini = min(denoised_array.min(), original_array.min(), noisy_array.min())

        # # # noisy_psnr, denoised_psnr = psnr(noisy_array, original_array, data_range=maxi -mini), psnr(denoised_array, original_array, data_range = maxi - mini)
        # # # noisy_ssim, denoised_ssim = ssim(noisy_array, original_array, data_range=maxi - mini), ssim(denoised_array, original_array, data_range=maxi - mini)
        # # # noisy_l1, denoised_l1 = np.mean(np.abs(noisy_array - original_array)), np.mean(np.abs(denoised_array - original_array))

        # # maxi = max((noisy_array - original_array).max(), (noisy_array - denoised_array).max())
        # # mini = min((noisy_array - original_array).min(), (noisy_array - denoised_array).min())

        # # psnr, denoised_psnr = psnr(noisy_array, original_array, data_range=maxi -mini), psnr(denoised_array, original_array, data_range = maxi - mini)
        # # # noisy_ssim, denoised_ssim = ssim(noisy_array, original_array, data_range=maxi - mini), ssim(denoised_array, original_array, data_range=maxi - mini)
        # # # noisy_l1, denoised_l1 = np.mean(np.abs(noisy_array - original_array)), np.mean(np.abs(denoised_array - original_array))

        # # # Print results
        # # print(f"Noisy PSNR: {noisy_psnr}, Denoised PSNR: {denoised_psnr}")
        # # print(f"Noisy SSIM: {noisy_ssim}, Denoised SSIM: {denoised_ssim}")
        # # print(f"Noisy L1: {noisy_l1}, Denoised L1: {denoised_l1}")
        # metrics = evaluate_denoising(original_array, noisy_array, denoised_array)


        # return metrics



        # Mask invalid values
        masked_array = ma.masked_invalid(denoised_array)
        matplotlib.image.imsave(f'{final_dir}/denoised_{file_name}.png', masked_array)

        # Load arrays
        original_array = np.load(save_dir.get('path'))[:, :, 1]
        noisy_array = np.load(save_dir.get('path'))[:, :, 0]

        # original_array = ((original_array - np.nanmin(original_array))/(np.nanmax(original_array) - np.nanmin(original_array)))
        # noisy_array = ((noisy_array - np.nanmin(noisy_array))/(np.nanmax(noisy_array) - np.nanmin(noisy_array)))
        # denoised_array = ((denoised_array - np.nanmin(denoised_array))/(np.nanmax(denoised_array) - np.nanmin(denoised_array)))

        # Create mask
        mask = ~np.isnan(denoised_array) & ~np.isnan(original_array) & ~np.isnan(noisy_array)

        # Convert to tensors
        # denoised_tensor = torch.tensor(denoised_array, dtype=torch.float32)
        # original_tensor = torch.tensor(original_array, dtype=torch.float32)
        # noisy_tensor = torch.tensor(noisy_array, dtype=torch.float32)

        # Apply mask
        denoised_array[~mask] = 0
        original_array[~mask] = 0
        noisy_array[~mask] = 0

        # # Reshape tensors
        # H, W = denoised_array.shape
        # denoised_tensor = denoised_tensor.view(1, 1, H, W)
        # original_tensor = original_tensor.view(1, 1, H, W)
        # noisy_tensor = noisy_tensor.view(1, 1, H, W)
        denoised_array = denoised_array.astype(np.float32)
        original_array = original_array.astype(np.float32)
        noisy_array = noisy_array.astype(np.float32)
        # Calculate PSNR, SSIM, L1

        maxi = max(denoised_array.max(), original_array.max(), noisy_array.max())
        mini = min(denoised_array.min(), original_array.min(), noisy_array.min())

        noisy_psnr, denoised_psnr = psnr(noisy_array, original_array, data_range=maxi -mini), psnr(denoised_array, original_array, data_range = maxi - mini)
        noisy_ssim, denoised_ssim = ssim(noisy_array, original_array, data_range=maxi - mini), ssim(denoised_array, original_array, data_range=maxi - mini)
        noisy_l1, denoised_l1 = np.mean(np.abs(noisy_array - original_array)), np.mean(np.abs(denoised_array - original_array))

        # Print results
        print(f"Noisy PSNR: {noisy_psnr}, Denoised PSNR: {denoised_psnr}")
        print(f"Noisy SSIM: {noisy_ssim}, Denoised SSIM: {denoised_ssim}")
        print(f"Noisy L1: {noisy_l1}, Denoised L1: {denoised_l1}")
        denoised_array[~mask] = np.nan
        original_array[~mask] = np.nan
        noisy_array[~mask] = np.nan

        return [denoised_array, original_array, noisy_array, [[noisy_psnr, denoised_psnr], [noisy_ssim, denoised_ssim], [noisy_l1, denoised_l1]]]


def trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        test_loader,
        reconstruct_loader,
        train_loss_fn: torch.nn.Module,
        val_loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        lr_scheduler_name: str,
        device: torch.device,
        epochs: int,
        save_dir: str,
        val_per_epoch: int,
        test_per_epoch: int,
        early_stopper=None,
        start_epoch = 1,
        test_dataset = None,
        train_dataset = None,
        val_dataset = None,
        pads = None,
        test_overlap = None,
        image_size = None,
        **kwargs
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
    }
    best_val_loss = 1e10
    if reconstruct_loader!=None:
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        torch.compile(model)
        model.eval()
        images = rec_step(model=model, test_dataset=test_dataset, reconstruct_loader=reconstruct_loader, device=device, save_dir=kwargs ,IMAGE_SIZE=image_size, pads = pads, test_overlap = test_overlap)
        return images

    for epoch in range(epochs + 1):
        print(f"Epoch {epoch}:")
        train_loss = train_step(model, train_loader, train_loss_fn, optimizer, device)
        results["train_loss"].append(train_loss)
        val_loss = val_step(model, val_loader, train_loader, val_loss_fn, device, epoch, val_per_epoch, save_dir)
        
        print()

        if lr_scheduler_name == "ReduceLROnPlateau":
            lr_scheduler.step(val_loss)
        elif lr_scheduler_name != "None":
            lr_scheduler.step()
        
        results["val_loss"].append(val_loss)
        
        checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler}
            

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, f"best_checkpoint.pth"))

        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))

        print(best_val_loss)
        # torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))

        if epoch%test_per_epoch == 0:
                checkpoint = torch.load(save_dir + "/best_checkpoint.pth")
                model.load_state_dict(checkpoint['model'])
                model.to(device)
                torch.compile(model)
                test_loss = test_step(model, test_loader, train_loader, val_loss_fn, device, epoch, test_per_epoch, save_dir)
                results["test_loss"].append(test_loss)

        torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

        if early_stopper is not None:
            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break





    return results