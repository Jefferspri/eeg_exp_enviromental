
# Third party moduls
from pygame import mixer
import _tkinter
import tkinter as tk
from tkinter import ttk
import datetime
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageSequence
import matplotlib.pyplot as plt
from scipy import stats
import random
from random import shuffle
from typing import List, Tuple
import sys
import threading
#from worker import create_worker,listen, sleep
import asyncio
import nest_asyncio
nest_asyncio.apply()
from async_tkinter_loop import async_handler, async_mainloop
from pylsl import StreamInlet, resolve_byprop
import math
from scipy import signal
from scipy.signal import filtfilt
import spkit as sp
import pywt
from sklearn import metrics
import pickle
import pydbus
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

# Train personalize class
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm


# own moduls
from moduls import tr_moduls as trm
from moduls.neurofeedback import record
from moduls import process_functions as pfunc

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the data from CSV files
df_raw = pd.read_csv("exports/eeg_995_M_25_A.csv")
df_details = pd.read_csv("exports/times_995_M_25_A.csv")

# Process the data
df_eeg, df_rt = pfunc.formating_data(df_raw, df_details)

print(df_rt['tr'].to_numpy())
print()
print()
print()

# Generate df_rt with date ranges
df_rt_date = pfunc.generate_df_rt_date_no_mean(df_rt)

print(df_rt_date['rt'].to_numpy())

plt.figure(figsize=(10, 6))
plt.plot(df_rt_date['rt'].to_numpy(), color='black', linewidth=2, label='TR')
plt.show()

# Interpolate values for trials without responses
df_rt_date = pfunc.interp_rt(df_rt_date)

# Compute VTC
df_rt_date = pfunc.compute_VTC(df_rt_date)
ori_med = df_rt_date['vtc'].median()

# Set classification labels
df_rt_date['class'] = np.where(df_rt_date['vtc'] >= ori_med, 0, 1)  # 0: out, 1: in

# Preprocess EEG data
df_eeg = pfunc.preprocessimg_data(df_eeg)

# Wavelet decomposition for feature extraction
df_features = pfunc.wavelet_packet_decomposition(df_eeg, df_rt_date)

# Normalize the features
df_features = pfunc.normalization(df_features)

# Add experiment number
df_features["n_experiment"] = 100

# Fill NaNs with 0
df_all_features = df_features.fillna(0)

# Train LGBM Regressor
str_tr_mean, mape, rmse, y_test, y_pred = pfunc.train_lgbm_regressor(df_all_features, "user_info")

# Display model performance
print(f"Model Performance: {str_tr_mean}, RMSE: {rmse:.4f} ms, MAPE: {mape:.4f}%")

# Train LGBM Classifier
class_y_test, class_y_pred = pfunc.train_lgbm_classifier(df_all_features, "user_info")

# Prepare data for plotting
x = np.arange(1, len(y_test) + 1)

# Plot TR prediction
plt.figure(figsize=(10, 6))
plt.scatter(x[:21], y_test[:21], color='#07D99A', label='Real TR')
plt.scatter(x[:21], y_pred[:21], color='#203ee6', marker='x', label='Predicted TR')
plt.plot(x[:21], class_y_test[:21], color='#07D99A')
plt.plot(x[:21], class_y_pred[:21], color='#203ee6')
plt.title('TR real vs TR predicho')
plt.xlabel('TR actual (seg.)')
plt.ylabel('TR predicho (seg.)')
plt.legend()
plt.grid(axis='x')
plt.show()

# Plot all clicks and power
plt.figure(figsize=(10, 6))
tr = df_all_features["rt"].to_numpy()
energy = df_all_features["AF8_total_energy"].to_numpy()
zcr = df_all_features["AF8_zcr"]
zcr_dir = df_all_features["AF8_zcr_dir"]

# Smooth data
tr_smooth = tr  # You can apply Gaussian smoothing here if needed
energy_smooth = gaussian_filter1d(energy, sigma=0.5)

xx = np.arange(0, len(df_all_features["vtc"]) * 0.8, 0.8)

# Create the bar chart
plt.plot(xx, df_all_features["vtc"] + 2, color='gray', linestyle='--', label='VTC')  # VTC
plt.plot(xx, df_all_features["class"] + 2, color='#4932DB', label='Class')  # Class
plt.plot(xx, 2 * (tr_smooth) + 1, color='black', linewidth=2, label='TR')  # TR
plt.plot(xx, energy_smooth + 1, label='Energy')
plt.plot(xx, zcr, label='ZCR')
plt.plot(xx, zcr_dir - 1, label='ZCR Direction')

# Vertical lines
for xline in [120, 176, 296, 352]:
    plt.axvline(x=xline, linewidth=2, color='gray', linestyle='--')

plt.title('TR y Características AF8 EEG ZCR')
plt.legend()
plt.grid(True)
plt.show()
