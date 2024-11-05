import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d


df_all_features = pd.read_csv("exports/all_features_jose___.csv")


# ----------------- PLOT 1

tr = df_all_features["rt"].to_numpy()
energy = df_all_features["AF8_total_energy"].to_numpy()
zcr = df_all_features["AF8_zcr"]
zcr_dir = df_all_features["AF8_zcr_dir"]

tr_smooth = tr#gaussian_filter1d(tr, sigma=0.5)
energy_smooth = energy#gaussian_filter1d(energy, sigma=0.5)

xx = np.linspace(0, 5.07, num=df_all_features["vtc"].shape[0])

ori_med = df_all_features["vtc"].median()

fig, ax = plt.subplots()

# create the barchart
plt.plot(xx, df_all_features["vtc"]+2, color= 'gray', linestyle="--") # real
plt.plot(xx, df_all_features["class"]+2, color= '#4932DB') # real
plt.plot(xx, 2*(tr_smooth)+1, color= 'black', linewidth=2) # real
plt.plot(xx,energy_smooth+1) 
plt.plot(xx,zcr) 
plt.plot(xx,zcr_dir-1)
plt.axvline(x=2, linewidth=2, color='gray', linestyle="--")
plt.axvline(x=3, linewidth=2, color='gray', linestyle="--")
plt.suptitle('TR y Características AF8 EEG ZCR')
plt.legend(['vtc','class','tr', 'energy', 'zcr', 'zcr_dir'])
plt.grid(True)
plt.show()


# ----------------- PLOT 2
p_delta = gaussian_filter1d(df_all_features["AF8_p_delta"], sigma=0.5)
p_theta = gaussian_filter1d(df_all_features["AF8_p_theta"], sigma=0.5)
p_alpha = gaussian_filter1d(df_all_features["AF8_p_alpha"], sigma=0.5)
p_beta = gaussian_filter1d(df_all_features["AF8_p_beta"], sigma=0.5)
p_gamma = gaussian_filter1d(df_all_features["AF8_p_gamma"], sigma=0.5)

fig, ax = plt.subplots()

# create the barchart
plt.plot(xx, df_all_features["vtc"]+2, color= 'gray', linestyle="--") # real
plt.plot(xx, df_all_features["class"]+2, color= '#4932DB') # real
plt.plot(xx, 2*(tr_smooth)+1, color= 'black', linewidth=2) # real
plt.plot(xx,p_delta+1) 
plt.plot(xx,p_theta) 
plt.plot(xx,p_alpha - 1) 
plt.plot(xx,p_beta - 2)
plt.plot(xx,p_gamma - 3)
plt.axvline(x=2, linewidth=2, color='gray', linestyle="--")
plt.axvline(x=3, linewidth=2, color='gray', linestyle="--")
plt.suptitle('TR y Características AF8 EEG Power')
plt.legend(['vtc','class','tr', 'P delta', 'P theta', 'P alpha', 'P beta', 'P gamma'])
plt.grid(True)
plt.show()

# ----------------- PLOT 3
af7_theta_std = gaussian_filter1d(df_all_features["AF7_std_theta"], sigma=0.5)
af7_beta_std = gaussian_filter1d(df_all_features["AF7_std_beta"], sigma=0.5)
af8_theta_std = gaussian_filter1d(df_all_features["AF8_std_theta"], sigma=0.5)
af8_beta_std = gaussian_filter1d(df_all_features["AF8_std_beta"], sigma=0.5)

fig, ax = plt.subplots()

# create the barchart
plt.plot(xx, df_all_features["vtc"]+2, color= 'gray', linestyle="--") # real
plt.plot(xx, df_all_features["class"]+2, color= '#4932DB') # real
plt.plot(xx, 2*(tr_smooth)+1, color= 'black', linewidth=2) # real
plt.plot(xx,af7_theta_std+1) 
plt.plot(xx,af7_beta_std) 
plt.plot(xx,af8_theta_std - 1) 
plt.plot(xx,af8_beta_std - 2)
plt.axvline(x=2, linewidth=2, color='gray', linestyle="--")
plt.axvline(x=3, linewidth=2, color='gray', linestyle="--")
plt.suptitle('TR y Características EEG STD')
plt.legend(['vtc','class','tr', 'AF7 theta std', 'AF7 beta std', 'AF8 theta std', 'AF8 beta std'])
plt.grid(True)
plt.show()


# ----------------- PLOT 4
ee = gaussian_filter1d(df_all_features["AF8_energy_entropy"], sigma=0.5)
se = gaussian_filter1d(df_all_features["AF8_spectral_entropy"], sigma=0.5)
sr = gaussian_filter1d(df_all_features["AF8_spectral_rollof"], sigma=0.5)
mfcc = gaussian_filter1d(df_all_features["AF8_mfcc"], sigma=0.5)

fig, ax = plt.subplots()

# create the barchart
plt.plot(xx, df_all_features["vtc"]+2, color= 'gray', linestyle="--") # real
plt.plot(xx, df_all_features["class"]+2, color= '#4932DB') # real
plt.plot(xx, 2*(tr_smooth)+1, color= 'black', linewidth=2) # real
plt.plot(xx, ee+1) 
plt.plot(xx, se) 
plt.plot(xx, sr - 1) 
plt.plot(xx,  mfcc - 2) 
plt.axvline(x=2, linewidth=2, color='gray', linestyle="--")
plt.axvline(x=3, linewidth=2, color='gray', linestyle="--")
plt.suptitle('TR y Características EEG Entropy')
plt.legend(['vtc','class','tr', 'Enery entropy', 'Spectral entropy', 'Spectral rollof', 'mfcc'])
plt.grid(True)
plt.show()