import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from utils import csv_to_numpy, plot_grid, preprocess_grid, split_grid
import os

IMAGE_SIZE = 256
PLOT_IMAGES = True
SAVE_DIR = '../images/'
path = '/home/santosh/Projects/geo_physics/meixia_noise_removal/MoEI_airborne_magdata/data/MAG_LEV_DGRF_surveyline_100m_cut.csv'
original_image = csv_to_numpy(path)
noisy_image = csv_to_numpy(path)

name = os.path.basename(path)
plot_grid(original_image, SAVE_DIR, name[:-4] + 'original_image')


in_out = np.dstack((noisy_image, original_image))
in_out_v2 = in_out.copy()
train_grid, test_grid, val_grid = split_grid(in_out_v2)
train_preprocessed,_ = preprocess_grid(train_grid, IMAGE_SIZE)
val_preprocessed ,_ = preprocess_grid(val_grid, IMAGE_SIZE)
test_preprocessed,_ = preprocess_grid(test_grid, IMAGE_SIZE)

if PLOT_IMAGES:
    plot_grid(train_preprocessed[:,:,0], SAVE_DIR, name[:-4] + 'train_image')
    plot_grid(val_preprocessed[:,:,0], SAVE_DIR, name[:-4] + 'val_image')
    plot_grid(test_preprocessed[:,:,0], SAVE_DIR, name[:-4] + 'test_image')