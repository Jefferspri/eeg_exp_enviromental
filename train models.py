
import pandas 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from moduls import process_functions as pfunc


df_all_features = pd.read_csv("exports/all_features_jose___.csv")

str_tr_mean, mape, rmse, y_test, y_pred = pfunc.train_lgbm_regressor(df_all_features, "jose")

print(str_tr_mean, mape, rmse)


class_y_test, class_y_pred = pfunc.train_lgbm_classifier(df_all_features, "jose")