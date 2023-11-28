# A File utilizing other classes and functions to plot and perform statistics of the models results (mostly hardcoded).
# This file is provided as a reference but links to models and csv files need to be corrected for it to run correctly.
# This is a part of Automating Hyperparameter Optimization in Geophysics with Optuna: A Comparative Study paper
# by Hussain Almarzooq & Umair Bin Waheed

import operator
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import dataGeneration as dg
import dataPreProcessing as dpp
from tcn import TCN
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

# denosing manual plots:
main_path = 'X\\'

df = pd.read_csv(main_path + "Results\Denoising\Search\GridSearch\\search_summary.csv")
# TCN Network:

# kernel size 1
plot_1_x_tcn_kernel = df["KRNL"].iloc[59:65].to_numpy()
plot_1_y_tcn_kernel = df["Validation Loss"].iloc[59:65].to_numpy()

# num filter 1
plot_1_x_tcn_filter = df["FIL"].iloc[52:59].to_numpy()
plot_1_y_tcn_filter = df["Validation Loss"].iloc[52:59].to_numpy()

# dilation 1
plot_1_x_tcn_dilation = df["DIL"].iloc[68:75].to_numpy()
plot_1_y_tcn_dilation = df["Validation Loss"].iloc[68:75].to_numpy()

# layer depth 1
plot_1_x_tcn_depth = df["LYR"].iloc[37:41].to_numpy()
plot_1_y_tcn_depth = df["Validation Loss"].iloc[37:41].to_numpy()

# num dnn 1
plot_1_x_tcn_dnn = df["DNN"].iloc[41:52].to_numpy()
plot_1_y_tcn_dnn = df["Validation Loss"].iloc[41:52].to_numpy()

# pooling 2
plot_2_x_tcn_pool = df["POL"].iloc[65:68].to_numpy()
plot_2_y_tcn_pool = df["Validation Loss"].iloc[65:68].to_numpy()

# pooling 2
plot_2_x_tcn_lr = [0.000001, 0.00001, 0.0001, 0.001]
plot_2_y_tcn_lr = df["Validation Loss"].iloc[75:79].to_numpy()

# CNN Network:

# kernel size 3
plot_3_x_cnn_kernel = df["KRNL"].iloc[23:29].to_numpy()
plot_3_y_cnn_kernel = df["Validation Loss"].iloc[23:29].to_numpy()

# num filter 3
plot_3_x_cnn_filter = df["FIL"].iloc[16:23].to_numpy()
plot_3_y_cnn_filter = df["Validation Loss"].iloc[16:23].to_numpy()

# num dnn 3
plot_3_x_cnn_dnn = df["DNN"].iloc[5:16].to_numpy()
plot_3_y_cnn_dnn = df["Validation Loss"].iloc[5:16].to_numpy()

# pooling 3
plot_3_x_cnn_pool = df["POL"].iloc[29:32].to_numpy()
plot_3_y_cnn_pool = df["Validation Loss"].iloc[29:32].to_numpy()

# layer depth 3
plot_3_x_cnn_depth = df["LYR"].iloc[1:5].to_numpy()
plot_3_y_cnn_depth = df["Validation Loss"].iloc[1:5].to_numpy()

# learning rate
plot_3_x_cnn_lr = [0.000001, 0.00001, 0.0001, 0.001]
plot_3_y_cnn_lr = df["Validation Loss"].iloc[32:36].to_numpy()

# CNN vs TCN
plot_4_x_min = df["Type"].iloc[[36, 0]].to_numpy()
plot_4_y_min = df["Validation Loss"].iloc[[36, 0]].to_numpy()
plot_4_path = main_path + 'Results\Denoising\Search\GridSearch\\type_val_csv.csv'
plot_4_data = csv.reader(open(plot_4_path), delimiter=',')
plot_4_data = sorted(plot_4_data, key=operator.itemgetter(0))
plot_4_data_plotting = np.array(plot_4_data)

# PLOTTING

# Plot 1
sns.set_theme()
title_font_size = 24
label_font_size = 12
fig_size_x = 24
fig_size_y = 8
f, ax = plt.subplots(nrows=2, ncols=4, figsize=(fig_size_x, fig_size_y), sharey=True)
f.suptitle('Conventional Grid Search for Denoising Model', fontsize=title_font_size)

ax[0, 0].set_xlabel('Network Type', fontsize=label_font_size)
ax[0, 0].set_ylabel('RMSE', fontsize=label_font_size)
sns.barplot(x=plot_4_x_min, y=plot_4_y_min, ax=ax[0, 0])

ax[0, 1].set_xlabel('Network Depth', fontsize=label_font_size)
# ax[0, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_1_x_tcn_depth, y=plot_1_y_tcn_depth, ax=ax[0, 1])
sns.scatterplot(x=plot_1_x_tcn_depth, y=plot_1_y_tcn_depth, ax=ax[0, 1])
sns.lineplot(x=plot_3_x_cnn_depth, y=plot_3_y_cnn_depth, ax=ax[0, 1], linewidth=3)
sns.scatterplot(x=plot_3_x_cnn_depth, y=plot_3_y_cnn_depth, ax=ax[0, 1])

ax[0, 2].set_xlabel('LOG(Number of Dense Neurons)', fontsize=label_font_size)
ax[0, 2].set_xscale('log')
# ax[0, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_1_x_tcn_dnn, y=plot_1_y_tcn_dnn, ax=ax[0, 2])
sns.scatterplot(x=plot_1_x_tcn_dnn, y=plot_1_y_tcn_dnn, ax=ax[0, 2])
sns.lineplot(x=plot_3_x_cnn_dnn, y=plot_3_y_cnn_dnn, ax=ax[0, 2], linewidth=3)
sns.scatterplot(x=plot_3_x_cnn_dnn, y=plot_3_y_cnn_dnn, ax=ax[0, 2])

ax[1, 0].set_xlabel('Number of Filters', fontsize=label_font_size)
ax[1, 0].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_1_x_tcn_filter, y=plot_1_y_tcn_filter, ax=ax[1, 0])
sns.scatterplot(x=plot_1_x_tcn_filter, y=plot_1_y_tcn_filter, ax=ax[1, 0])
sns.lineplot(x=plot_3_x_cnn_filter, y=plot_3_y_cnn_filter, ax=ax[1, 0], linewidth=3)
sns.scatterplot(x=plot_3_x_cnn_filter, y=plot_3_y_cnn_filter, ax=ax[1, 0])

ax[1, 1].set_xlabel('Kernel Size', fontsize=label_font_size)
# ax[1, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_1_x_tcn_kernel, y=plot_1_y_tcn_kernel, ax=ax[1, 1])
sns.scatterplot(x=plot_1_x_tcn_kernel, y=plot_1_y_tcn_kernel, ax=ax[1, 1])
sns.lineplot(x=plot_3_x_cnn_kernel, y=plot_3_y_cnn_kernel, ax=ax[1, 1], linewidth=3)
sns.scatterplot(x=plot_3_x_cnn_kernel, y=plot_3_y_cnn_kernel, ax=ax[1, 1])

ax[1, 2].set_xlabel('Max Pooling Value', fontsize=label_font_size)
# ax[1, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_2_x_tcn_pool, y=plot_2_y_tcn_pool, ax=ax[1, 2])
sns.scatterplot(x=plot_2_x_tcn_pool, y=plot_2_y_tcn_pool, ax=ax[1, 2])
sns.lineplot(x=plot_3_x_cnn_pool, y=plot_3_y_cnn_pool, ax=ax[1, 2], linewidth=3)
sns.scatterplot(x=plot_3_x_cnn_pool, y=plot_3_y_cnn_pool, ax=ax[1, 2])

ax[0, 3].set_xlabel('LOG(Learning Rate)', fontsize=label_font_size)
ax[0, 3].set_xscale('log')
# ax[1, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_2_x_tcn_lr, y=plot_2_y_tcn_lr, ax=ax[0, 3])
sns.scatterplot(x=plot_2_x_tcn_lr, y=plot_2_y_tcn_lr, ax=ax[0, 3])
sns.lineplot(x=plot_3_x_cnn_lr, y=plot_3_y_cnn_lr, ax=ax[0, 3], linewidth=3)
sns.scatterplot(x=plot_3_x_cnn_lr, y=plot_3_y_cnn_lr, ax=ax[0, 3])

ax[1, 3].set_xlabel('Network Type', fontsize=label_font_size)
sns.boxplot(x=plot_4_data_plotting[:, 0], y=np.genfromtxt(plot_4_data_plotting[:, 1]), order=['TCN', 'CNN'],
            ax=ax[1, 3])

# f.delaxes(ax[1, 3])
plt.ylim(0, 0.30)

plt.show()

# inversion manual plots:

df2 = pd.read_csv(main_path + "Results\Inversion\Search\GridSearch\\search_summary.csv")

# Network Comparison Plot 5:
plot_5_x_network = df2["Type"].iloc[[136, 0, 33, 69, 102]].to_numpy()
plot_5_y_network = df2["Validation Loss"].iloc[[136, 0, 33, 69, 102]].to_numpy()

# Network Depth Plot 6:
plot_6_x_depth_dnn = df2["LYR"].iloc[103:107].to_numpy()
plot_6_y_depth_dnn = df2["Validation Loss"].iloc[103:107].to_numpy()

plot_6_x_depth_cnn = df2["LYR"].iloc[1:5].to_numpy()
plot_6_y_depth_cnn = df2["Validation Loss"].iloc[1:5].to_numpy()

plot_6_x_depth_cnn_var = df2["LYR"].iloc[70:74].to_numpy()
plot_6_y_depth_cnn_var = df2["Validation Loss"].iloc[70:74].to_numpy()

plot_6_x_depth_cnn_lstm = df2["LYR"].iloc[34:37].to_numpy()
plot_6_y_depth_cnn_lstm = df2["Validation Loss"].iloc[34:37].to_numpy()

plot_6_x_depth_tcn = df2["LYR"].iloc[137:141].to_numpy()
plot_6_y_depth_tcn = df2["Validation Loss"].iloc[137:141].to_numpy()

# Network Dense Plot 7:
plot_7_x_dnn_dnn = df2["DNN"].iloc[107:118].to_numpy()
plot_7_y_dnn_dnn = df2["Validation Loss"].iloc[107:118].to_numpy()

plot_7_x_dnn_cnn = df2["DNN"].iloc[5:16].to_numpy()
plot_7_y_dnn_cnn = df2["Validation Loss"].iloc[5:16].to_numpy()

plot_7_x_dnn_cnn_var = df2["DNN"].iloc[74:85].to_numpy()
plot_7_y_dnn_cnn_var = df2["Validation Loss"].iloc[74:85].to_numpy()

plot_7_x_dnn_cnn_lstm = df2["DNN"].iloc[37:48].to_numpy()
plot_7_y_dnn_cnn_lstm = df2["Validation Loss"].iloc[37:48].to_numpy()

plot_7_x_dnn_tcn = df2["DNN"].iloc[141:152].to_numpy()
plot_7_y_dnn_tcn = df2["Validation Loss"].iloc[141:152].to_numpy()

# Network Filters Plot 8:

plot_8_x_filt_cnn = df2["FIL"].iloc[16:23].to_numpy()
plot_8_y_filt_cnn = df2["Validation Loss"].iloc[16:23].to_numpy()

plot_8_x_filt_cnn_var = df2["FIL"].iloc[85:92].to_numpy()
plot_8_y_filt_cnn_var = df2["Validation Loss"].iloc[85:92].to_numpy()

plot_8_x_filt_cnn_lstm = df2["FIL"].iloc[48:55].to_numpy()
plot_8_y_filt_cnn_lstm = df2["Validation Loss"].iloc[48:55].to_numpy()

plot_8_x_filt_tcn = df2["FIL"].iloc[152:159].to_numpy()
plot_8_y_filt_tcn = df2["Validation Loss"].iloc[152:159].to_numpy()

# Network Kernel Size Plot 9:

plot_9_x_kernel_cnn = df2["KRNL"].iloc[23:29].to_numpy()
plot_9_y_kernel_cnn = df2["Validation Loss"].iloc[23:29].to_numpy()

plot_9_x_kernel_cnn_var = df2["KRNL"].iloc[92:98].to_numpy()
plot_9_y_kernel_cnn_var = df2["Validation Loss"].iloc[92:98].to_numpy()

plot_9_x_kernel_cnn_lstm = df2["KRNL"].iloc[55:61].to_numpy()
plot_9_y_kernel_cnn_lstm = df2["Validation Loss"].iloc[55:61].to_numpy()

plot_9_x_kernel_tcn = df2["KRNL"].iloc[159:165].to_numpy()
plot_9_y_kernel_tcn = df2["Validation Loss"].iloc[159:165].to_numpy()

# Network Dilations & LSTM Plot X (TCN Only):
plot_10_x_dil_tcn = df2["DIL"].iloc[165:172].to_numpy()
plot_10_y_dil_tcn = df2["Validation Loss"].iloc[165:172].to_numpy()

plot_10_x_units_cnn_lstm = [5, 10, 25, 50]
plot_10_y_units_cnn_lstm = df2["Validation Loss"].iloc[61:65].to_numpy()

# Network LR Plot 7:
plot_11_x_lr_dnn = [0.000001, 0.00001, 0.0001, 0.001]
plot_11_y_lr_dnn = df2["Validation Loss"].iloc[132:136].to_numpy()

plot_11_x_lr_cnn = [0.000001, 0.00001, 0.0001, 0.001]
plot_11_y_lr_cnn = df2["Validation Loss"].iloc[29:33].to_numpy()

plot_11_x_lr_cnn_var = [0.000001, 0.00001, 0.0001, 0.001]
plot_11_y_lr_cnn_var = df2["Validation Loss"].iloc[98:102].to_numpy()

plot_11_x_lr_cnn_lstm = [0.000001, 0.00001, 0.0001, 0.001]
plot_11_y_lr_cnn_lstm = df2["Validation Loss"].iloc[65:69].to_numpy()

plot_11_x_lr_tcn = [0.000001, 0.00001, 0.0001, 0.001]
plot_11_y_lr_tcn = df2["Validation Loss"].iloc[172:176].to_numpy()

# CNN_LSTM vs TCN
plot_11_path = main_path + 'Results\Inversion\Search\GridSearch\\type_val_csv.csv'
plot_11_data = csv.reader(open(plot_4_path), delimiter=',')
plot_11_data = sorted(plot_11_data, key=operator.itemgetter(0))
plot_11_data_plotting = np.array(plot_11_data)

# Plot 3
sns.set_theme()
f, ax = plt.subplots(nrows=2, ncols=4, figsize=(fig_size_x, fig_size_y), sharey=True)
f.suptitle('Conventional Grid Search for Velocity Inversion Model', fontsize=title_font_size)

ax[0, 0].set_xlabel('Network Type', fontsize=label_font_size)
ax[0, 0].set_ylabel('RMSE', fontsize=label_font_size)
sns.barplot(x=plot_5_x_network, y=plot_5_y_network, ax=ax[0, 0])

ax[0, 1].set_xlabel('Network Depth', fontsize=label_font_size)
# ax[0, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_6_x_depth_tcn, y=plot_6_y_depth_tcn, ax=ax[0, 1], linewidth=3)
sns.scatterplot(x=plot_6_x_depth_tcn, y=plot_6_y_depth_tcn, ax=ax[0, 1])
sns.lineplot(x=plot_6_x_depth_cnn, y=plot_6_y_depth_cnn, ax=ax[0, 1], linestyle='--')
sns.scatterplot(x=plot_6_x_depth_cnn, y=plot_6_y_depth_cnn, ax=ax[0, 1])
sns.lineplot(x=plot_6_x_depth_cnn_lstm, y=plot_6_y_depth_cnn_lstm, ax=ax[0, 1])
sns.scatterplot(x=plot_6_x_depth_cnn_lstm, y=plot_6_y_depth_cnn_lstm, ax=ax[0, 1])
sns.lineplot(x=plot_6_x_depth_cnn_var, y=plot_6_y_depth_cnn_var, ax=ax[0, 1])
sns.scatterplot(x=plot_6_x_depth_cnn_var, y=plot_6_y_depth_cnn_var, ax=ax[0, 1])
sns.lineplot(x=plot_6_x_depth_dnn, y=plot_6_y_depth_dnn, ax=ax[0, 1], linestyle='--')
sns.scatterplot(x=plot_6_x_depth_dnn, y=plot_6_y_depth_dnn, ax=ax[0, 1])

ax[0, 2].set_xlabel('LOG(Number of Dense Neurons)', fontsize=label_font_size)
ax[0, 2].set_xscale('log')
# ax[0, 2].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_7_x_dnn_tcn, y=plot_7_y_dnn_tcn, ax=ax[0, 2], linewidth=3)
sns.scatterplot(x=plot_7_x_dnn_tcn, y=plot_7_y_dnn_tcn, ax=ax[0, 2])
sns.lineplot(x=plot_7_x_dnn_cnn, y=plot_7_y_dnn_cnn, ax=ax[0, 2], linestyle='--')
sns.scatterplot(x=plot_7_x_dnn_cnn, y=plot_7_y_dnn_cnn, ax=ax[0, 2])
sns.lineplot(x=plot_7_x_dnn_cnn_lstm, y=plot_7_y_dnn_cnn_lstm, ax=ax[0, 2])
sns.scatterplot(x=plot_7_x_dnn_cnn_lstm, y=plot_7_y_dnn_cnn_lstm, ax=ax[0, 2])
sns.lineplot(x=plot_7_x_dnn_cnn_var, y=plot_7_y_dnn_cnn_var, ax=ax[0, 2])
sns.scatterplot(x=plot_7_x_dnn_cnn_var, y=plot_7_y_dnn_cnn_var, ax=ax[0, 2])
sns.lineplot(x=plot_7_x_dnn_dnn, y=plot_7_y_dnn_dnn, ax=ax[0, 2], linestyle='--')
sns.scatterplot(x=plot_7_x_dnn_dnn, y=plot_7_y_dnn_dnn, ax=ax[0, 2])

ax[1, 0].set_xlabel('Number of Filters', fontsize=label_font_size)
ax[1, 0].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_8_x_filt_tcn, y=plot_8_y_filt_tcn, ax=ax[1, 0], linewidth=3)
sns.scatterplot(x=plot_8_x_filt_tcn, y=plot_8_y_filt_tcn, ax=ax[1, 0])
sns.lineplot(x=plot_8_x_filt_cnn, y=plot_8_y_filt_cnn, ax=ax[1, 0], linestyle='--')
sns.scatterplot(x=plot_8_x_filt_cnn, y=plot_8_y_filt_cnn, ax=ax[1, 0])
sns.lineplot(x=plot_8_x_filt_cnn_lstm, y=plot_8_y_filt_cnn_lstm, ax=ax[1, 0])
sns.scatterplot(x=plot_8_x_filt_cnn_lstm, y=plot_8_y_filt_cnn_lstm, ax=ax[1, 0])
sns.lineplot(x=plot_8_x_filt_cnn_var, y=plot_8_y_filt_cnn_var, ax=ax[1, 0])
sns.scatterplot(x=plot_8_x_filt_cnn_var, y=plot_8_y_filt_cnn_var, ax=ax[1, 0])

ax[1, 1].set_xlabel('Kernel Size', fontsize=label_font_size)
# ax[1, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_9_x_kernel_tcn, y=plot_9_y_kernel_tcn, ax=ax[1, 1], linewidth=3)
sns.scatterplot(x=plot_9_x_kernel_tcn, y=plot_9_y_kernel_tcn, ax=ax[1, 1])
sns.lineplot(x=plot_9_x_kernel_cnn, y=plot_9_y_kernel_cnn, ax=ax[1, 1], linestyle='--')
sns.scatterplot(x=plot_9_x_kernel_cnn, y=plot_9_y_kernel_cnn, ax=ax[1, 1])
sns.lineplot(x=plot_9_x_kernel_cnn_lstm, y=plot_9_y_kernel_cnn_lstm, ax=ax[1, 1])
sns.scatterplot(x=plot_9_x_kernel_cnn_lstm, y=plot_9_y_kernel_cnn_lstm, ax=ax[1, 1])
sns.lineplot(x=plot_9_x_kernel_cnn_var, y=plot_9_y_kernel_cnn_var, ax=ax[1, 1])
sns.scatterplot(x=plot_9_x_kernel_cnn_var, y=plot_9_y_kernel_cnn_var, ax=ax[1, 1])

ax[1, 2].set_xlabel('Dilations (TCN) / LSTM Units (CNN-LSTM)', fontsize=label_font_size)
# ax[1, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_10_x_dil_tcn, y=plot_10_y_dil_tcn, ax=ax[1, 2], linewidth=3)
sns.scatterplot(x=plot_10_x_dil_tcn, y=plot_10_y_dil_tcn, ax=ax[1, 2])
sns.lineplot(x=plot_10_x_units_cnn_lstm, y=plot_10_y_units_cnn_lstm, ax=ax[1, 2])
sns.scatterplot(x=plot_10_x_units_cnn_lstm, y=plot_10_y_units_cnn_lstm, ax=ax[1, 2])

ax[0, 3].set_xlabel('LOG(Learning Rate)', fontsize=label_font_size)
ax[0, 3].set_xscale('log')
# ax[0, 1].set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=plot_11_x_lr_tcn, y=plot_11_y_lr_tcn, ax=ax[0, 3], linewidth=3)
sns.scatterplot(x=plot_11_x_lr_tcn, y=plot_11_y_lr_tcn, ax=ax[0, 3])
sns.lineplot(x=plot_11_x_lr_cnn, y=plot_11_y_lr_cnn, ax=ax[0, 3], linestyle='--')
sns.scatterplot(x=plot_11_x_lr_cnn, y=plot_11_y_lr_cnn, ax=ax[0, 3])
sns.lineplot(x=plot_11_x_lr_cnn_lstm, y=plot_11_y_lr_cnn_lstm, ax=ax[0, 3])
sns.scatterplot(x=plot_11_x_lr_cnn_lstm, y=plot_11_y_lr_cnn_lstm, ax=ax[0, 3])
sns.lineplot(x=plot_11_x_lr_cnn_var, y=plot_11_y_lr_cnn_var, ax=ax[0, 3])
sns.scatterplot(x=plot_11_x_lr_cnn_var, y=plot_11_y_lr_cnn_var, ax=ax[0, 3])
sns.lineplot(x=plot_11_x_lr_dnn, y=plot_11_y_lr_dnn, ax=ax[0, 3], linestyle='--')
sns.scatterplot(x=plot_11_x_lr_dnn, y=plot_11_y_lr_dnn, ax=ax[0, 3])

ax[1, 3].set_xlabel('Network Type', fontsize=label_font_size)
sns.boxplot(x=plot_11_data_plotting[:, 0], y=np.genfromtxt(plot_11_data_plotting[:, 1]),
            order=['TCN', 'CNN', 'CNN_LSTM', 'CNN_VAR', 'DNN'],
            ax=ax[1, 3])

# f.delaxes(ax[1, 3])

plt.ylim(0, 15000)

plt.show()

# Final Models Plotting

df3 = pd.read_csv(main_path + "Results\Denoising\Repeatability\\repeatability_summary.csv")

den_x_app = df3["Approach"].iloc[:].to_numpy()
den_y_val = df3["Validation Loss"].iloc[:].to_numpy()

df4 = pd.read_csv(main_path + "Results\Inversion\Repeatability\\repeatability_summary.csv")

inv_x_app = df4["Approach"].iloc[:].to_numpy()
inv_y_val = df4["Validation Loss"].iloc[:].to_numpy()

# Plot 4
sns.set_theme()
title_font_size = 22
label_font_size = 14
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=False)
f.suptitle('Final Models Search', fontsize=title_font_size)

ax[0].set_xlabel('Denoising Problem', fontsize=label_font_size)
sns.boxplot(x=den_x_app, y=den_y_val, order=['Grid Search', 'Optuna'], ax=ax[0])
ax[0].set_ylim([0, 0.15])
ax[0].set_ylabel('RMSE', fontsize=label_font_size)
ax[1].set_xlabel('Inversion Problem', fontsize=label_font_size)
sns.boxplot(x=inv_x_app, y=inv_y_val, order=['Grid Search', 'Optuna'], ax=ax[1])
ax[1].set_ylim([1500, 4000])
ax[1].set_ylabel('RMSE', fontsize=label_font_size)

plt.show()

# Optuna Plotting
optuna_plot = 0
if optuna_plot == 1:
    study_name_inv = 'INV_Study_20231023_0642_a230862c-ae7f-4982-a193-428836215a2f'
    storage_name_inv = 'sqlite:///Results\Inversion\Search\Optuna\INV_Study_20231023_0642_a230862c-ae7f-4982-a193-428836215a2f_Final.db'

    study_inv = optuna.create_study(study_name=study_name_inv, direction="minimize",
                                    storage=storage_name_inv, load_if_exists=True)

    fig = plot_optimization_history(study_inv)
    fig.show()
    fig = plot_parallel_coordinate(study_inv)
    fig.show()
    fig = plot_contour(study_inv)
    fig.show()
    fig = plot_slice(study_inv)
    fig.show()
    fig = plot_param_importances(study_inv)
    fig.show()
    fig = optuna.visualization.plot_param_importances(
        study_inv, target=lambda t: t.duration.total_seconds(), target_name="duration"
    )
    fig.show()
    fig = plot_edf(study_inv)
    fig.show()

    study_name_den = 'DEN_Study_20231016_1758_aa36c319-e052-452e-8e1c-9d6ad4c02885'
    storage_name_den = 'sqlite:///Results\Denoising\Search\Optuna\DEN_Study_20231016_1758_aa36c319-e052-452e-8e1c-9d6ad4c02885_Final.db'

    study_den = optuna.create_study(study_name=study_name_den, direction="minimize",
                                    storage=storage_name_den, load_if_exists=True)

    fig = plot_optimization_history(study_den)
    fig.show()
    fig = plot_parallel_coordinate(study_den)
    fig.show()
    fig = plot_contour(study_den)
    fig.show()
    fig = plot_slice(study_den)
    fig.show()
    fig = plot_param_importances(study_den)
    fig.show()
    fig = optuna.visualization.plot_param_importances(
        study_den, target=lambda t: t.duration.total_seconds(), target_name="duration"
    )
    fig.show()
    fig = plot_edf(study_den)
    fig.show()

# Generators used for datasets.
stats_calc = 1
data_num = 1000
dgc_predict_datasets = dg.DataGenerationClass(save_data_to_file=0, run_number=data_num)
dgc_predict_datasets.run_data_generation(gen_num_start=0, gen_num_end=data_num)

# Inversion Plots
ml_model_tcn_1 = keras.models.load_model(
    main_path + 'Results\Inversion\Final_Model\GridSearch\\TCN_SEQ_LYR1_DNN3072_DIL16_FIL128_KRNL64_POL1_LSTM0_NA_20231031_2221_best_model.h5',
    custom_objects={"root_mean_squared_error": dpp.root_mean_squared_error, "TCN": TCN})
ml_model_tcn_2 = keras.models.load_model(
    main_path + 'Results\Inversion\Final_Model\Optuna\\TCN_SEQ_LYR1_DNN3824_DIL108_FIL46_KRNL14_POL1_LSTM0_NA_20231031_2302_best_model.h5',
    custom_objects={"root_mean_squared_error": dpp.root_mean_squared_error, "TCN": TCN})

# --- Retrieving one series
dgc_predict = dg.DataGenerationClass(save_data_to_file=0)
dgc_predict.generate_dataset()
x, y = dgc_predict.retrieve_mult_input_output(x_data='CONV_REFL_CLEAN', y_data='VEL')

y_hat_tcn_1 = ml_model_tcn_1.predict(x)[0, :, 0]
y_hat_tcn_2 = ml_model_tcn_2.predict(x)[0, :, 0]

x = np.squeeze(x[0, :])
y = np.squeeze(y[0, :])
t = np.linspace(0, 4.096, 4096)

diff_tcn_1 = np.sqrt(np.mean(np.square(y - y_hat_tcn_1)))
diff_tcn_2 = np.sqrt(np.mean(np.square(y - y_hat_tcn_2)))

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
f.suptitle('Inversion Results Comparsion', fontsize=title_font_size)

ax[0].set_xlabel('Time (s)', fontsize=label_font_size)
ax[0].set_ylabel('Velocity (ft/s)', fontsize=label_font_size)
sns.lineplot(x=t, y=y, ax=ax[0], label='Actual Velocity')
sns.lineplot(x=t, y=y_hat_tcn_1, ax=ax[0], label='Grid Search Model Prediction')
sns.lineplot(x=t, y=y_hat_tcn_2, ax=ax[0], label='Optuna Model Prediction')
ax[0].set_ylim([5000, 30000])

ax[1].set_xlabel('Time (s)', fontsize=label_font_size)
ax[1].set_ylabel('Velocity Difference (ft/s)', fontsize=label_font_size)
sns.lineplot(x=t, y=y - y, ax=ax[1], label='Actual=0')
sns.lineplot(x=t, y=y - y_hat_tcn_1, ax=ax[1], label='Grid Search RMSE=' + str(np.round(diff_tcn_1)))
sns.lineplot(x=t, y=y - y_hat_tcn_2, ax=ax[1], label='Optuna RMSE=' + str(np.round(diff_tcn_2)))
ax[1].set_ylim([-5000, 5000])

plt.show()

# Plot 3
grid_search_results_path = main_path + 'Results\Inversion\Final_Model\GridSearch\\TCN_SEQ_LYR1_DNN3072_DIL16_FIL128_KRNL64_POL1_LSTM0_NA_20231031_2221_log.csv'
grid_search_results = csv.reader(open(grid_search_results_path), delimiter=',')

optuna_results_path = main_path + 'Results\Inversion\Final_Model\Optuna\\TCN_SEQ_LYR1_DNN3824_DIL108_FIL46_KRNL14_POL1_LSTM0_NA_20231031_2302_log.csv'
optuna_results = csv.reader(open(optuna_results_path), delimiter=',')

grid_search_results = sorted(grid_search_results, key=operator.itemgetter(0))
optuna_results = sorted(optuna_results, key=operator.itemgetter(0))
grid_search_results_plotting = np.array(grid_search_results)
optuna_results_plotting = np.array(optuna_results)

sns.set_theme()
title_font_size = 36
label_font_size = 18
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
f.suptitle('Loss History for Grid Search versus Optuna', fontsize=title_font_size)

ax.set_xlabel('Epoch', fontsize=label_font_size)
ax.set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=np.genfromtxt(grid_search_results_plotting[1:, 0]), y=np.genfromtxt(grid_search_results_plotting[1:, 3]),
             ax=ax, label='Grid Search', color='orange')
sns.lineplot(x=np.genfromtxt(optuna_results_plotting[1:, 0]), y=np.genfromtxt(optuna_results_plotting[1:, 3]), ax=ax,
             label='Optuna', color='green')

plt.xlim(0, 100)
plt.ylim(1000, 5000)

plt.show()

# Denoising Plots:
ae_model_cnn_1 = keras.models.load_model(
    main_path + 'Results\Denoising\Final_Model\GridSearch\\CNN_AE_LYR1_DNN2_DIL128_FIL64_KRNL8_POL2_LSTM0_NA_20231102_0100_best_model.h5',
    custom_objects={"root_mean_squared_error": dpp.root_mean_squared_error})
ae_model_cnn_2 = keras.models.load_model(
    main_path + 'Results\Denoising\Final_Model\Optuna\\CNN_AE_LYR2_DNN1792_DIL128_FIL64_KRNL12_POL2_LSTM0_NA_20231102_0038_best_model.h5',
    custom_objects={"root_mean_squared_error": dpp.root_mean_squared_error})

# --- Retrieving one series
x, y = dgc_predict.retrieve_mult_input_output(x_data='CONV_REFL_NOISY', y_data='CONV_REFL_CLEAN')

y_model_cnn_1 = ae_model_cnn_1.predict(x)[0, :, 0]
y_model_cnn_2 = ae_model_cnn_2.predict(x)[0, :, 0]

x = np.squeeze(x[0, :])
y = np.squeeze(y[0, :])
t = np.linspace(0, 4.096, 4096)

diff_cnn_1 = np.sqrt(np.mean(np.square(y - y_model_cnn_1)))  # np.mean(np.abs((y - y_model_cnn_1) / y))
diff_cnn_2 = np.sqrt(np.mean(np.square(y - y_model_cnn_2)))  # np.mean(np.abs((y - y_model_cnn_2) / y))

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
f.suptitle('Inversion Results Comparsion', fontsize=title_font_size)

ax[0].set_xlabel('Time (s)', fontsize=label_font_size)
ax[0].set_ylabel('Amplitude', fontsize=label_font_size)
sns.lineplot(x=t, y=y, ax=ax[0], label='Actual Clean Trace')
sns.lineplot(x=t, y=y_model_cnn_1, ax=ax[0], label='Grid Search Model Prediction')
sns.lineplot(x=t, y=y_model_cnn_2, ax=ax[0], label='Optuna Model Prediction')
ax[0].set_ylim([-0.5, 0.5])

ax[1].set_xlabel('Time (s)', fontsize=label_font_size)
ax[1].set_ylabel('Amplitude Difference', fontsize=label_font_size)
sns.lineplot(x=t, y=y - y, ax=ax[1], label='Actual=0')
sns.lineplot(x=t, y=y - y_model_cnn_1, ax=ax[1], label='Grid Search RMSE=' + str(np.round(diff_cnn_1, 4)))
sns.lineplot(x=t, y=y - y_model_cnn_2, ax=ax[1], label='Optuna RMSE=' + str(np.round(diff_cnn_2, 4)))
ax[1].set_ylim([-0.5, 0.5])

plt.show()

# Plot 5
grid_search_results_path = main_path + 'Results\Denoising\Final_Model\GridSearch\\CNN_AE_LYR1_DNN2_DIL128_FIL64_KRNL8_POL2_LSTM0_NA_20231102_0100_log.csv'
grid_search_results = csv.reader(open(grid_search_results_path), delimiter=',')

optuna_results_path = main_path + 'Results\Denoising\Final_Model\Optuna\\CNN_AE_LYR2_DNN1792_DIL128_FIL64_KRNL12_POL2_LSTM0_NA_20231102_0038_log.csv'
optuna_results = csv.reader(open(optuna_results_path), delimiter=',')
grid_search_results = sorted(grid_search_results, key=operator.itemgetter(0))
optuna_results = sorted(optuna_results, key=operator.itemgetter(0))
grid_search_results_plotting = np.array(grid_search_results)
optuna_results_plotting = np.array(optuna_results)

sns.set_theme()
title_font_size = 36
label_font_size = 18
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
f.suptitle('Loss History for Grid Search versus Optuna', fontsize=title_font_size)

ax.set_xlabel('Epoch', fontsize=label_font_size)
ax.set_ylabel('RMSE', fontsize=label_font_size)
sns.lineplot(x=np.genfromtxt(grid_search_results_plotting[1:, 0]), y=np.genfromtxt(grid_search_results_plotting[1:, 3]),
             ax=ax, label='Grid Search', color='orange')
sns.lineplot(x=np.genfromtxt(optuna_results_plotting[1:, 0]), y=np.genfromtxt(optuna_results_plotting[1:, 3]), ax=ax,
             label='Optuna', color='green')

plt.xlim(0, 50)
plt.ylim(0.015, 0.05)

plt.show()

if stats_calc == 1:
    x_datasets, y_datasets = dgc_predict_datasets.retrieve_mult_input_output(k_start=0, k_end=data_num,
                                                                             x_data='CONV_REFL_NOISY',
                                                                             y_data='CONV_REFL_CLEAN')

    y_hat_cnn_2 = ae_model_cnn_2.predict(x_datasets)[:, :, 0]
    y_hat_cnn_1 = ae_model_cnn_1.predict(x_datasets)[:, :, 0]

    diff_cnn_1 = np.sqrt(np.mean(np.square(y_datasets - y_hat_cnn_1)))
    diff_cnn_2 = np.sqrt(np.mean(np.square(y_datasets - y_hat_cnn_2)))

    print('Grid Search Denoising model average RMSE is ' + str(diff_cnn_1))
    print('Optuna Denoising model average RMSE is ' + str(diff_cnn_2))


if stats_calc == 1:
    x_datasets, y_datasets = dgc_predict_datasets.retrieve_mult_input_output(k_start=0, k_end=data_num,
                                                                             x_data='CONV_REFL_CLEAN',
                                                                             y_data='VEL')

    y_hat_tcn_1 = ml_model_tcn_1.predict(x_datasets)[:, :, 0]
    y_hat_tcn_2 = ml_model_tcn_2.predict(x_datasets)[:, :, 0]

    diff_tcn_1 = np.sqrt(np.mean(np.square(y_datasets - y_hat_tcn_1)))
    diff_tcn_2 = np.sqrt(np.mean(np.square(y_datasets - y_hat_tcn_2)))

    print('Grid Search Inversion model average RMSE is ' + str(diff_tcn_1))
    print('Optuna Inversion model average RMSE is ' + str(diff_tcn_2))
