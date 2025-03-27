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


def PSNR(preds, targets):
     psnr = ignite.metrics.PSNR(data_range = targets.max() - targets.min())
     psnr.update((preds, targets))
     return psnr.compute()
def SSIM(preds, targets):
     psnr = ignite.metrics.SSIM(data_range = targets.max() - targets.min())
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

    pred_psnr, noisy_psnr, pred_ssim, noisy_ssim, pred_l1, noisy_l1 = [], [], [], [], [], []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(reconstruct_loader), desc=f"Test denoising", total=len(reconstruct_loader)):
            # noisy_patches, original_patches, minis, maxis, width, height = divide_and_resize_v2(data)
            # noisy_patches = noisy_patches.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
            # original_patches = original_patches.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
            noisy_patches, original_patches, [mini_noisy, mini_clean, maxi_noisy, maxi_clean], [width, height] = data
            noisy_patches = noisy_patches.float().to(device)
            original_patches = original_patches.float().to(device)
            outputs = model(noisy_patches).to(device)
            


            outputs = (outputs.float().to(device) * (maxi_noisy.float().to(device) - mini_noisy.float().to(device))) + mini_noisy.float().to(device)
            # noisy_patches = (noisy_patches.float().to(device) * (maxi_noisy.float().to(device) - mini_noisy.float().to(device))) + mini_noisy.float().to(device)
            # original_patches = (original_patches.float().to(device) * (maxi_clean.float().to(device) - mini_clean.float().to(device))) + mini_clean.float().to(device)

            noisy_patches = torch.where(noisy_patches == 0, 0, (noisy_patches.float().to(device) * (maxi_noisy.float().to(device) - mini_noisy.float().to(device)) + mini_noisy.float().to(device))) 
            original_patches = torch.where(original_patches == 0, 0, (original_patches.float().to(device) * (maxi_clean.float().to(device) - mini_clean.float().to(device)) + mini_clean.float().to(device))) 

            list_of_denoised_patches.append(outputs.cpu())
            list_of_noisy_patches.append(noisy_patches.cpu())
            list_of_original_patches.append(original_patches.cpu())
            minis_list.append(mini_clean.float().to(device))
            maxis_list.append(maxi_clean.float().to(device))


            pred_psnr.append(PSNR(outputs, original_patches))
            noisy_psnr.append(PSNR(noisy_patches, original_patches))

            pred_ssim.append(SSIM(outputs, original_patches))
            noisy_ssim.append(SSIM(noisy_patches, original_patches))

            pred_l1.append(L1(outputs, original_patches))
            noisy_l1.append(L1(noisy_patches, original_patches))


        denoised_stack = torch.cat(list_of_denoised_patches, dim=0)
        noisy_stack = torch.cat(list_of_noisy_patches, dim=0)
        original_stack = torch.cat(list_of_original_patches, dim=0)

        pred_psnr, noisy_psnr = sum(pred_psnr)/len(pred_psnr), sum(noisy_psnr)/len(noisy_psnr)
        pred_ssim, noisy_ssim = sum(pred_ssim)/len(pred_ssim), sum(noisy_ssim)/len(noisy_ssim)
        pred_l1, noisy_l1 = sum(pred_l1)/len(pred_l1), sum(noisy_l1)/len(noisy_l1)

        breakpoint()
        # num_images = 20
        # # indices = torch.randperm(denoised_stack.size(0))
        # noisy_patches_subset = noisy_stack[indices]
        # outputs_subset = denoised_stack[indices]
        # original_patches_subset = original_stack[indices]


        save_path = os.path.join(save_dir, f'epoch_test_test_denoise.png')
        # IMAGE_SIZE = 128
        # save_subplot(noisy_patches_subset, outputs_subset, original_patches_subset, save_path, num_images)
        denoised_image = merge_patches_with_median(denoised_stack, IMAGE_SIZE, [width, height], IMAGE_SIZE - IMAGE_SIZE//test_overlap, mode='median')
        matplotlib.image.imsave(f'{save_dir}/denoised_airmag_recent.png', denoised_image)
        original_image = merge_patches_with_median(original_stack, IMAGE_SIZE, [width, height], IMAGE_SIZE - IMAGE_SIZE//test_overlap, mode='median')
        matplotlib.image.imsave(f'{save_dir}/original_airmag_recent.png', original_image)
        noisy_image = merge_patches_with_median(noisy_stack, IMAGE_SIZE, [width, height], IMAGE_SIZE - IMAGE_SIZE//test_overlap, mode='median')
        matplotlib.image.imsave(f'{save_dir}/noisy_airmag_recent.png', noisy_image)
        original_array = np.where(original_image == 0, np.nan, original_image)
        noisy_array = np.where(noisy_image == 0, np.nan, noisy_image)
        denoised_array = np.where(np.isnan(test_dataset.padded_data[:,:,1]), np.nan, denoised_image)
        # return [denoised_array[5:-5, 1:-1], original_array[5:-5, 1:-1], noisy_array[5:-5,1:-1]]
        print(denoised_array.shape)
        if test_dataset.padding_info[0][1]!=0:
            if test_dataset.padding_info[0][3]!=0:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                original_array = original_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                noisy_array = noisy_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
            else:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:]
                original_array = original_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:]
                noisy_array = noisy_array[test_dataset.padding_info[0][0]:-test_dataset.padding_info[0][1], test_dataset.padding_info[0][2]:]
        else:
            if test_dataset.padding_info[0][3]!=0:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                original_array = original_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
                noisy_array = noisy_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:-test_dataset.padding_info[0][3]]
            else:
                denoised_array = denoised_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:]
                original_array = original_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:]
                noisy_array = noisy_array[test_dataset.padding_info[0][0]:, test_dataset.padding_info[0][2]:]



        mask = ~np.isnan(denoised_array) & ~np.isnan(original_array) & ~np.isnan(noisy_array)

        denoised_tensor = torch.tensor(denoised_array, dtype=torch.float32)
        original_tensor = torch.tensor(original_array, dtype=torch.float32)
        noisy_tensor = torch.tensor(noisy_array, dtype=torch.float32)

        # Mask out NaNs
        denoised_tensor[~mask] = 0
        original_tensor[~mask] = 0
        noisy_tensor[~mask] = 0

        # Ensure shape is (B, C, H, W) - You need the original image size
        H, W = denoised_array.shape  # Replace with your actual image size

        # Reshape to 4D tensor (Batch, Channels, Height, Width)
        denoised_tensor = denoised_tensor.view(1, 1, H, W)
        original_tensor = original_tensor.view(1, 1, H, W)
        noisy_tensor = noisy_tensor.view(1, 1, H, W)
        print(f"Noisy PSNR: {PSNR(noisy_tensor, original_tensor)}, Denoised PSNR: {PSNR(denoised_tensor, original_tensor)}")
        print(f"Noisy SSIM: {SSIM(noisy_tensor, original_tensor)}, Denoised SSIM: {SSIM(denoised_tensor, original_tensor)}")
        print(f"Noisy L1: {L1(noisy_tensor, original_tensor)}, Denoised L1: {L1(denoised_tensor, original_tensor)}")

        breakpoint()    


        return [denoised_array, original_array, noisy_array]


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
        image_size = None

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
        images = rec_step(model=model, test_dataset=test_dataset, reconstruct_loader=reconstruct_loader, device=device, save_dir=os.path.dirname(save_dir) ,IMAGE_SIZE=image_size, pads = pads, test_overlap = test_overlap)
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