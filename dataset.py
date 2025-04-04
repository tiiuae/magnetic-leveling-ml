
import torch
import math
import random
import numpy as np
from torch.utils.data import Dataset

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
 
START_seed()


class BrazilDatasetPretraining(Dataset):
    def __init__(self, datasets, patch_dim=256, stride=64, transform=None, fill_value=0, split = 'train'):
        """
        Args:
        - datasets (list): List containing the data arrays.
        - patch_dim (int): The size of each patch.
        - stride (int): The step size or overlap between patches.
        - transform (callable, optional): A function/transform to apply to the patches.
        - fill_value (float): Value to fill NaN or padded regions.
        - seed (int): Random seed for reproducibility.
        """
        super().__init__()

        self.patch_size = patch_dim
        self.stride = stride
        self.transform = transform
        self.fill_value = fill_value

        self.preprocessed_datasets = []
        self.padding_info = []
        self.scaling_info = []
        self.dataset_patches = []
        self.split = split
        for filepath in datasets:
            data = np.load(filepath)
            # print(data.shape)
            processed_data, self.padded_data, scale_info, pad_info = self.preprocess_grid(data, self.patch_size, self.stride, fill=self.fill_value)
            self.preprocessed_datasets.append(processed_data)
            self.scaling_info.append(scale_info)
            self.padding_info.append(pad_info)
            num_patches = self.calculate_num_patches(processed_data.shape)
            self.dataset_patches.append(num_patches)
            print(processed_data.shape)

        
        self.total_patches = sum(self.dataset_patches)
        # if self.split != 'test':
        # #     np.random.seed(self.seed)
        self.indices = np.arange(self.total_patches)
        # # if self.split != 'test':
        # #     np.random.shuffle(self.indices)

    def replace_nan_with_edge_values(self, arr):
        # Get the shape of the array
        rows, cols, channels = arr.shape

        # Replace NaNs in rows and then in columns for each channel
        for c in range(channels):
            # Replace NaNs in rows
            for i in range(rows):
                first_valid_value = None
                # Find the first non-NaN value in the row for the current channel
                for value in arr[i, :, c]:
                    if not np.isnan(value):
                        first_valid_value = value
                        break
                
                # Replace NaNs with the first valid value in the row
                if first_valid_value is not None:
                    arr[i, np.isnan(arr[i, :, c]), c] = first_valid_value

            # Replace NaNs in columns
            for j in range(cols):
                first_valid_value = None
                # Find the first non-NaN value in the column for the current channel
                for i in range(rows):
                    if not np.isnan(arr[i, j, c]):
                        first_valid_value = arr[i, j, c]
                        break
                
                # Replace NaNs with the first valid value in the column
                if first_valid_value is not None:
                    arr[np.isnan(arr[:, j, c]), j, c] = first_valid_value

        return arr

    def preprocess_grid(self, grid, image_size, stride, fill=None):
        """
        Preprocess a 3D grid by normalizing, filling NaNs, and padding to a fixed size.

        Args:
        - grid (np.ndarray): 3D numpy array of shape (H, W, C).
        - image_size (int): Target patch size.
        - stride (int): Stride for overlapping patches.
        - fill (float): Value to fill NaNs and pad regions.
        
        Returns:
        - grid_v2 (np.ndarray): Preprocessed grid.
        - scale_info (list): Min and max values for normalization.
        - pad_info (list): Padding information for all sides.
        """
        mini, maxi = np.nanmin(grid[:, :, 0]), np.nanmax(grid[:, :, 0])

        x, y, z = grid.shape[0], grid.shape[1], 2

        # padded_array = np.zeros((x + 256, y + 256, z))

        # # Copy the original array into the center
        # padded_array[128:-128, 128:-128, :] = grid

        # # Step 2: Reflect the rows for top and bottom padding
        # padded_array[0:128, 128:-128, :] = grid[0:128, :, :][::-1, :, :]  # Top
        # padded_array[-128:, 128:-128, :] = grid[-128:, :, :][::-1, :, :]  # Bottom

        # # Step 3: Reflect the columns for left and right padding
        # padded_array[128:-128, 0:128, :] = grid[:, 0:128, :][:, ::-1, :]  # Left
        # padded_array[128:-128, -128:, :] = grid[:, -128:, :][:, ::-1, :]  # Right

        # # Step 4: Reflect diagonally for corners
        # padded_array[0:128, 0:128, :] = grid[0:128, 0:128, :][::-1, ::-1, :]  # Top-left
        # padded_array[0:128, -128:, :] = grid[0:128, -128:, :][::-1, ::-1, :]  # Top-right
        # padded_array[-128:, 0:128, :] = grid[-128:, 0:128, :][::-1, ::-1, :]  # Bottom-left
        # padded_array[-128:, -128:, :] = grid[-128:, -128:, :][::-1, ::-1, :]  # Bottom-right
        padded_array = grid







        # Adjust padding to ensure the grid can be split into patches with given stride
        new_width = math.ceil((padded_array.shape[0] - image_size) / stride) * stride + image_size
        new_height = math.ceil((padded_array.shape[1] - image_size) / stride) * stride + image_size

        pad_left = (new_width - padded_array.shape[0]) // 2
        pad_right = new_width - padded_array.shape[0] - pad_left
        pad_top = (new_height - padded_array.shape[1]) // 2
        pad_bottom = new_height - padded_array.shape[1] - pad_top
        # breakpoint()
        # grid_v1 = self.replace_nan_with_edge_values(grid)
        grid_v1 = np.pad(padded_array, pad_width=((pad_left, pad_right), (pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=np.nan)
        # breakpoint()
        grid_v2 = np.nan_to_num(grid_v1, nan=fill)
        # print(grid.shape, grid_v1.shape, grid_v2.shape)
        return grid_v2, grid_v1, [mini, maxi], [pad_left, pad_right, pad_top, pad_bottom]

    def calculate_num_patches(self, shape):
        """
        Calculate the number of patches that can be extracted from a grid of given shape.
        """
        patches_x = (shape[0] - self.patch_size) // self.stride + 1
        patches_y = (shape[1] - self.patch_size) // self.stride + 1
        return patches_x * patches_y


    def random_crop(self, array, sizes=[128, 256, 512]):
        assert array.shape[0] >= 128 and array.shape[1] >= 128, "Array is too small for cropping"
        
        crop_size = random.choice(sizes)
        if crop_size == 512:
            return array
        max_x = array.shape[0] - crop_size
        max_y = array.shape[1] - crop_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return array[x:x+crop_size, y:y+crop_size]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        dataset_idx = 0
        cumulative_patches = 0
        for i, num_patches in enumerate(self.dataset_patches):
            if actual_index < cumulative_patches + num_patches:
                dataset_idx = i
                break
            cumulative_patches += num_patches

        data = self.preprocessed_datasets[dataset_idx]
        shape = data.shape
        width = data.shape[1]
        height = data.shape[0]
        patch_index = actual_index - cumulative_patches

        patches_per_row = (shape[1] - self.patch_size) // self.stride + 1
        row = (patch_index // patches_per_row) * self.stride
        col = (patch_index % patches_per_row) * self.stride

        data = data[row:row + self.patch_size, col:col + self.patch_size,:]

        noisy_patch = data[:, :, 0]
        clean_patch = data[:,:, 1]


        try:
            non_zero_vals = noisy_patch[noisy_patch != 0]
            noisy_patch = np.where(noisy_patch == 0, 0, (noisy_patch - np.min(non_zero_vals)) / (np.max(non_zero_vals) - np.min(non_zero_vals) + 1e-6))
            mini_noisy = np.min(non_zero_vals)
            maxi_noisy = np.max(non_zero_vals)

            non_zero_vals = clean_patch[clean_patch != 0]
            clean_patch = np.where(clean_patch == 0, 0, (clean_patch - np.min(non_zero_vals)) / (np.max(non_zero_vals) - np.min(non_zero_vals) + 1e-6))
            mini_clean = np.min(non_zero_vals)
            maxi_clean = np.max(non_zero_vals)

        except:
            noisy_patch = noisy_patch
            clean_patch = clean_patch
            mini_noisy = 0
            maxi_noisy = 0
            mini_clean = 0
            maxi_clean = 0

        
        if self.transform:
            state = torch.get_rng_state()
            noisy_patch = self.transform(noisy_patch)            
            torch.set_rng_state(state)
            clean_patch = self.transform(clean_patch)
        return noisy_patch, noisy_patch, [mini_noisy, mini_clean, maxi_noisy, maxi_clean], [width, height]
    



class BrazilDatasetFinetuning(Dataset):
    def __init__(self, datasets, patch_dim=256, stride=64, transform=None, fill_value=0, split = 'train'):
        """
        Args:
        - datasets (list): List containing the data arrays.
        - patch_dim (int): The size of each patch.
        - stride (int): The step size or overlap between patches.
        - transform (callable, optional): A function/transform to apply to the patches.
        - fill_value (float): Value to fill NaN or padded regions.
        - seed (int): Random seed for reproducibility.
        """
        super().__init__()

        self.patch_size = patch_dim
        self.stride = stride
        self.transform = transform
        self.fill_value = fill_value

        self.preprocessed_datasets = []
        self.padding_info = []
        self.scaling_info = []
        self.dataset_patches = []
        self.split = split
        for filepath in datasets:
            data = np.load(filepath)
            processed_data, self.padded_data, scale_info, pad_info = self.preprocess_grid(data, self.patch_size, self.stride, fill=self.fill_value)
            self.preprocessed_datasets.append(processed_data)
            self.scaling_info.append(scale_info)
            self.padding_info.append(pad_info)
            num_patches = self.calculate_num_patches(processed_data.shape)
            self.dataset_patches.append(num_patches)
            print(processed_data.shape)

        
        self.total_patches = sum(self.dataset_patches)
        # if self.split != 'test':
        # #     np.random.seed(self.seed)
        self.indices = np.arange(self.total_patches)
        # # if self.split != 'test':
        # #     np.random.shuffle(self.indices)

    def replace_nan_with_edge_values(self, arr):
        # Get the shape of the array
        rows, cols, channels = arr.shape

        # Replace NaNs in rows and then in columns for each channel
        for c in range(channels):
            # Replace NaNs in rows
            for i in range(rows):
                first_valid_value = None
                # Find the first non-NaN value in the row for the current channel
                for value in arr[i, :, c]:
                    if not np.isnan(value):
                        first_valid_value = value
                        break
                
                # Replace NaNs with the first valid value in the row
                if first_valid_value is not None:
                    arr[i, np.isnan(arr[i, :, c]), c] = first_valid_value

            # Replace NaNs in columns
            for j in range(cols):
                first_valid_value = None
                # Find the first non-NaN value in the column for the current channel
                for i in range(rows):
                    if not np.isnan(arr[i, j, c]):
                        first_valid_value = arr[i, j, c]
                        break
                
                # Replace NaNs with the first valid value in the column
                if first_valid_value is not None:
                    arr[np.isnan(arr[:, j, c]), j, c] = first_valid_value

        return arr

    def preprocess_grid(self, grid, image_size, stride, fill=None):
        """
        Preprocess a 3D grid by normalizing, filling NaNs, and padding to a fixed size.

        Args:
        - grid (np.ndarray): 3D numpy array of shape (H, W, C).
        - image_size (int): Target patch size.
        - stride (int): Stride for overlapping patches.
        - fill (float): Value to fill NaNs and pad regions.
        
        Returns:
        - grid_v2 (np.ndarray): Preprocessed grid.
        - scale_info (list): Min and max values for normalization.
        - pad_info (list): Padding information for all sides.
        """
        mini, maxi = np.nanmin(grid[:, :, 0]), np.nanmax(grid[:, :, 0])

        x, y, z = grid.shape[0], grid.shape[1], 2

        # padded_array = np.zeros((x + 256, y + 256, z))

        # # Copy the original array into the center
        # padded_array[128:-128, 128:-128, :] = grid

        # # Step 2: Reflect the rows for top and bottom padding
        # padded_array[0:128, 128:-128, :] = grid[0:128, :, :][::-1, :, :]  # Top
        # padded_array[-128:, 128:-128, :] = grid[-128:, :, :][::-1, :, :]  # Bottom

        # # Step 3: Reflect the columns for left and right padding
        # padded_array[128:-128, 0:128, :] = grid[:, 0:128, :][:, ::-1, :]  # Left
        # padded_array[128:-128, -128:, :] = grid[:, -128:, :][:, ::-1, :]  # Right

        # # Step 4: Reflect diagonally for corners
        # padded_array[0:128, 0:128, :] = grid[0:128, 0:128, :][::-1, ::-1, :]  # Top-left
        # padded_array[0:128, -128:, :] = grid[0:128, -128:, :][::-1, ::-1, :]  # Top-right
        # padded_array[-128:, 0:128, :] = grid[-128:, 0:128, :][::-1, ::-1, :]  # Bottom-left
        # padded_array[-128:, -128:, :] = grid[-128:, -128:, :][::-1, ::-1, :]  # Bottom-right
        padded_array = grid







        # Adjust padding to ensure the grid can be split into patches with given stride
        new_width = max(math.ceil((padded_array.shape[0] - image_size) / stride) * stride,0) + image_size
        new_height = max(math.ceil((padded_array.shape[1] - image_size) / stride),0) * stride + image_size

        pad_left = (new_width - padded_array.shape[0]) // 2
        pad_right = new_width - padded_array.shape[0] - pad_left
        pad_top = (new_height - padded_array.shape[1]) // 2
        pad_bottom = new_height - padded_array.shape[1] - pad_top
        # breakpoint()
        # grid_v1 = self.replace_nan_with_edge_values(grid)
        grid_v1 = np.pad(padded_array, pad_width=((pad_left, pad_right), (pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=np.nan)
        # breakpoint()
        grid_v2 = np.nan_to_num(grid_v1, nan=fill)
        # print(grid.shape, grid_v1.shape, grid_v2.shape)
        pad_left = (new_width - grid.shape[0]) // 2
        pad_right = new_width - grid.shape[0] - pad_left
        pad_top = (new_height - grid.shape[1]) // 2
        pad_bottom = new_height - grid.shape[1] - pad_top
        return grid_v2, grid_v1, [mini, maxi], [pad_left, pad_right, pad_top, pad_bottom]

    def calculate_num_patches(self, shape):
        """
        Calculate the number of patches that can be extracted from a grid of given shape.
        """
        patches_x = (shape[0] - self.patch_size) // self.stride + 1
        patches_y = (shape[1] - self.patch_size) // self.stride + 1
        return patches_x * patches_y


    def random_crop(self, array, sizes=[128, 256, 512]):
        assert array.shape[0] >= 128 and array.shape[1] >= 128, "Array is too small for cropping"
        
        crop_size = random.choice(sizes)
        if crop_size == 512:
            return array
        max_x = array.shape[0] - crop_size
        max_y = array.shape[1] - crop_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return array[x:x+crop_size, y:y+crop_size]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        dataset_idx = 0
        cumulative_patches = 0
        for i, num_patches in enumerate(self.dataset_patches):
            if actual_index < cumulative_patches + num_patches:
                dataset_idx = i
                break
            cumulative_patches += num_patches

        data = self.preprocessed_datasets[dataset_idx]
        shape = data.shape
        width = data.shape[1]
        height = data.shape[0]
        patch_index = actual_index - cumulative_patches

        patches_per_row = (shape[1] - self.patch_size) // self.stride + 1
        row = (patch_index // patches_per_row) * self.stride
        col = (patch_index % patches_per_row) * self.stride

        data = data[row:row + self.patch_size, col:col + self.patch_size,:]

        noisy_patch = data[:, :, 0]
        clean_patch = data[:,:, 1]


        try:
            non_zero_vals = noisy_patch[noisy_patch != 0]
            noisy_patch = np.where(noisy_patch == 0, 0, (noisy_patch - np.min(non_zero_vals)) / (np.max(non_zero_vals) - np.min(non_zero_vals) + 1e-6))
            mini_noisy = np.min(non_zero_vals)
            maxi_noisy = np.max(non_zero_vals)

            non_zero_vals = clean_patch[clean_patch != 0]
            clean_patch = np.where(clean_patch == 0, 0, (clean_patch - np.min(non_zero_vals)) / (np.max(non_zero_vals) - np.min(non_zero_vals) + 1e-6))
            mini_clean = np.min(non_zero_vals)
            maxi_clean = np.max(non_zero_vals)

        except:
            noisy_patch = noisy_patch
            clean_patch = clean_patch
            mini_noisy = 0
            maxi_noisy = 0
            mini_clean = 0
            maxi_clean = 0

        
        if self.transform:
            state = torch.get_rng_state()
            noisy_patch = self.transform(noisy_patch)            
            torch.set_rng_state(state)
            clean_patch = self.transform(clean_patch)
        return noisy_patch, clean_patch, [mini_noisy, mini_clean, maxi_noisy, maxi_clean], [width, height]