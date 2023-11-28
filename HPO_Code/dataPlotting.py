# Data Plotting for the Lookahead VSP Project. It includes functions that deal with common plots for this project.
# This is a part of Automating Hyperparameter Optimization in Geophysics with Optuna: A Comparative Study paper
# by Hussain Almarzooq & Umair Bin Waheed

# Import required libraries:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras
from nptyping import NDArray
from scipy import interpolate
from IPython.display import clear_output
from keras import callbacks


# Define Font Sizes
def font_sizes_defaults(small_size: int, medium_size: int, bigger_size: int):
    """
    Function that handles font sizes for plots.
    Arguments:
        small_size (int): value for small font size.
        medium_size (int): value for medium font size.
        bigger_size (int): value for large font size.
    Returns:
        None.
    """
    plt.rc('font', size=medium_size)  # controls default text sizes
    plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title


# Max value for axis:
def max_axis_value(data) -> float:
    """
    Function that retrieves the absolute maximum value of a dataset.
    Arguments:
        data: dataset to retireve max value.
    Returns:
        Absolute maximum value of the dataset.
    """
    max_value = np.abs(np.max(data))

    return max_value


# Plot layers & model related plots
def plot_model(lithology_type: list[int], depth_layer: list[int], vel_layer: list[int], dens_layer: list[int],
               imp_layer: list[int]):
    """
    Function that plots lithology, velocity, density, and impedance in depth.
    Arguments:
        lithology_type (list[int]): List of lithologies.
        depth_layer (list[int]): List of depths.
        vel_layer (list[int]): List of velocities.
        dens_layer (list[int]): List of densities.
        imp_layer (list[int]): List of impedance values.
    Returns:
        None, plots the model.
    """
    # Interpolation for continuous depth
    depth_cont = np.arange(0, np.max(depth_layer))
    litho_interp = interpolate.interp1d(depth_layer, lithology_type, kind='previous', fill_value='extrapolate')
    vel_interp = interpolate.interp1d(depth_layer, vel_layer, kind='previous', fill_value='extrapolate')
    dens_interp = interpolate.interp1d(depth_layer, dens_layer, kind='previous', fill_value='extrapolate')
    imp_interp = interpolate.interp1d(depth_layer, imp_layer, kind='previous', fill_value='extrapolate')

    # Interpolated values
    litho_cont = litho_interp(depth_cont)
    vel_cont = vel_interp(depth_cont)
    dens_cont = dens_interp(depth_cont)
    imp_cont = imp_interp(depth_cont)

    # Initialize Plots

    # Lithology Key:
    lithology_numbers = {0: {'lith': 'Sandstone', 'lith_num': 0, 'hatch': '..', 'color': '#ffff00'},
                         1: {'lith': 'Shale', 'lith_num': 1, 'hatch': '--', 'color': '#bebebe'},
                         2: {'lith': 'Limestone', 'lith_num': 2, 'hatch': '+', 'color': '#80ffff'},
                         3: {'lith': 'Dolomite', 'lith_num': 3, 'hatch': '-/', 'color': '#8080ff'},
                         4: {'lith': 'Anhydrite', 'lith_num': 4, 'hatch': '', 'color': '#ff80ff'},
                         5: {'lith': 'Halite', 'lith_num': 5, 'hatch': 'x', 'color': '#7ddfbe'}}

    # Create dataframe with lithologies
    df_lith = pd.DataFrame.from_dict(lithology_numbers, orient='index')
    df_lith.index.name = 'LITHOLOGY'

    # Font Size:
    font_sizes_defaults(small_size=8, medium_size=16, bigger_size=24)

    # Global Plot Parameters
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey='row', figsize=(10, 10))
    fig.gca().invert_yaxis()
    fig.suptitle('Model Plot')
    plt.ylim([np.max(depth_cont), 0])

    # Lithology Plot:
    ax1.step(litho_cont, depth_cont, color="black", linewidth=0.5)
    ax1.title.set_text('Lithology')
    ax1.set_ylabel('Depth (ft)')
    ax1.set_xlim(0, 1)
    ax1.xaxis.label.set_color("black")
    ax1.tick_params(axis='x', colors="black")
    ax1.spines["top"].set_edgecolor("black")

    for key in lithology_numbers.keys():
        color = lithology_numbers[key]['color']
        hatch = lithology_numbers[key]['hatch']
        ax1.fill_betweenx(depth_cont, 0, 1, where=(litho_cont == key),
                          facecolor=color, hatch=hatch)

    ax1.xaxis.set_visible(False)

    # Velocity Plot
    ax2.plot(vel_cont, depth_cont, color='tab:purple')
    ax2.title.set_text('Velocity')
    ax2.set_xlabel('Velocity (ft/s)')
    ax2.set_xlim([5000.0, 25000.0])

    # Density Plot
    ax3.plot(dens_cont, depth_cont, color='tab:red')
    ax3.title.set_text('Density')
    ax3.set_xlabel('Density (g/cm^3)')
    ax3.set_xlim([1.5, 3])

    # Impedance Plot
    ax4.plot(imp_cont, depth_cont, color='black')
    ax4.title.set_text('Impedance')
    ax4.set_xlabel('Impedance (Pa.s/ft)')
    ax4.set_xlim([10000.0, 70000.0])

    fig.show()


# Cross plot of velocity versus density colored by lithology, sometimes  the crossplot lithology colors produce wrong colors, replotting helps,
# # cause of the bug is unclear.
def plot_xplot(lith_time: NDArray, vel_time: NDArray, dens_time: NDArray, imp_time: NDArray = (-999 * np.ones(1))):
    """
    Function that crossplots the velocity with the density colored by lithology or impedance if supplied.
    Arguments:
        lith_time (NDArray): Lithology array in time.
        vel_time (NDArray): Velocity array in time.
        dens_time (NDArray): Density array in time.
        imp_time (NDArray) (opt): Impedance array in time, optional.
    Returns:
        None, crossplots the model.
    """
    # Determine lithology indices
    sandstone = np.where(lith_time == 0.)[0]
    if len(sandstone) == 0:
        sandstone = np.where(lith_time == 0)[0]
    shale = np.where(lith_time == 1)[0]
    limestone = np.where(lith_time == 2)[0]
    dolomite = np.where(lith_time == 3)[0]
    anhydrite = np.where(lith_time == 4)[0]
    halite = np.where(lith_time == 5)[0]

    # Font Size:
    font_sizes_defaults(small_size=12, medium_size=16, bigger_size=24)

    plt.subplots(1, figsize=(10, 10))
    plt.xlim([5000, 25000])
    plt.ylim([1.9, 3])
    plt.xlabel('Velocity (ft)')
    plt.ylabel('Density (g/cm^3)')

    if imp_time[0] == -999:
        plt.suptitle('Velocity versus Density Cross Plot')
        plt.scatter(vel_time[sandstone], dens_time[sandstone], c="y", alpha=0.5)
        plt.scatter(vel_time[shale], dens_time[shale], c="tab:gray", alpha=0.5)
        plt.scatter(vel_time[limestone], dens_time[limestone], c="tab:cyan", alpha=0.5)
        plt.scatter(vel_time[dolomite], dens_time[dolomite], c="tab:purple", alpha=0.5)
        plt.scatter(vel_time[anhydrite], dens_time[anhydrite], c="tab:pink", alpha=0.5)
        plt.scatter(vel_time[halite], dens_time[halite], c="xkcd:lime", alpha=0.5)

        plt.legend(["Sandstone", "Shale", "Limestone", "Dolomite", "Anhydrite", "Halite"])
    else:
        plt.suptitle('Velocity versus Density Cross Plot colored by Impedance')
        ax = sns.scatterplot(x=vel_time, y=dens_time, hue=imp_time, alpha=0.5, palette='vlag')

        norm = plt.Normalize(10000, 70000)
        sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        ax.figure.colorbar(sm)

    plt.show()


# Plot reflectivity and other generated data:
def plot_refl(t: NDArray, refl_time: NDArray, refl_time_wavelet: NDArray, refl_time_wavelet_noise: NDArray,
              refl_time_wavelet_noise_gaps: NDArray, title_comment: str = ''):
    """
    Function that plots the different stages of reflectivity series creation in time.
    Arguments:
        t (NDArray): Array of time values.
        refl_time (NDArray): Raw reflectivity series.
        refl_time_wavelet (NDArray): Convolved reflectivity series.
        refl_time_wavelet_noise (NDArray): Reflectivity series with noise added.
        refl_time_wavelet_noise_gaps (NDArray): Reflectivity series with noise and gaps added.
        title_comment (str): String to add to the title of the plot.
    Returns:
        None, plots the different stages of reflectivity series.
    """
    # Font Size:
    font_sizes_defaults(small_size=12, medium_size=16, bigger_size=24)

    max_value = max_axis_value([np.max(refl_time), np.max(refl_time_wavelet), np.max(refl_time_wavelet_noise),
                                np.max(refl_time_wavelet_noise_gaps)])

    # Global Plot Parameters
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey='row', figsize=(10, 10))
    fig.gca().invert_yaxis()
    fig.suptitle('Data Plot, Wavelet = ' + title_comment)
    plt.ylim([np.max(t), 0])
    plt.xlim([-max_value - 0.2, max_value + 0.2])

    # Reflectivty Data Plot
    ax1.plot(refl_time / max_value, t, c='k')
    ax1.title.set_text('Reflectivty Series')
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel('Two-Way Time (s)')

    # Convolved Data Plot
    ax2.plot(refl_time_wavelet / max_value, t, c='k')
    ax2.title.set_text('Convolved Data')
    ax2.xaxis.set_visible(False)

    # Convolved Data Plot with Noise
    ax3.plot(refl_time_wavelet_noise / max_value, t, c='k')
    ax3.title.set_text('Noise Added')
    ax3.xaxis.set_visible(False)

    # Convolved Data Plot with Noise and Gaps
    ax4.plot(refl_time_wavelet_noise_gaps / max_value, t, c='k')
    ax4.title.set_text('Gap Added')
    ax4.xaxis.set_visible(False)

    fig.show()


# Plot comparison between wavelets:
def plot_refl_compare(t, refl_time, refl_time_ricker, refl_time_klauder, refl_time_orsmby):
    """
    Function that plots the different the reflectivity series with different wavelets convolved.
    Arguments:
        t (NDArray): Array of time values.
        refl_time (NDArray): Raw reflectivity series.
        refl_time_ricker (NDArray): Convolved reflectivity series with ricker wavelet.
        refl_time_klauder (NDArray): Convolved reflectivity series with klauder wavelet.
        refl_time_orsmby (NDArray): Convolved reflectivity series with orsmby wavelet.
    Returns:
        None, plots the reflectivity with different wavelets.
    """
    # Font Size:
    font_sizes_defaults(small_size=6, medium_size=8, bigger_size=16)

    max_value = max_axis_value([np.max(refl_time), np.max(refl_time_ricker), np.max(refl_time_klauder),
                                np.max(refl_time_orsmby)])

    # Global Plot Parameters
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey='row', figsize=(10, 10))
    fig.gca().invert_yaxis()
    fig.suptitle('Data Plot Comparison')
    plt.ylim([np.max(t), 0])
    plt.xlim([-max_value - 0.2, max_value + 0.2])

    # Reflectivty Data Plot
    ax1.plot(refl_time / max_value, t, c='k')
    ax1.title.set_text('Reflectivty Series')
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel('Two-Way Time (s)')

    # Convolved Data Plot
    ax2.plot(refl_time_ricker / max_value, t, c='k')
    ax2.title.set_text('Ricker Data')
    ax2.xaxis.set_visible(False)

    # Convolved Data Plot with Noise
    ax3.plot(refl_time_klauder / max_value, t, c='k')
    ax3.title.set_text('Klauder Added')
    ax3.xaxis.set_visible(False)

    # Convolved Data Plot with Noise and Gaps
    ax4.plot(refl_time_orsmby / max_value, t, c='k')
    ax4.title.set_text('Orsmby Added')
    ax4.xaxis.set_visible(False)

    fig.show()


# Plot AE prediction results
def plot_ae_predict_raw(x: NDArray, x_hat: NDArray, save_path: str = ''):
    """
    Function that plots actual and predicted (reconstructed) AE result.
    Arguments:
        x: Actual series.
        x_hat: Predicted series.
        save_path: path to save file, leave empty if file saving is not desired.
    Returns:
        None, plots actual and predicted (reconstructed) AE result.
    """

    # Font Size:
    font_sizes_defaults(small_size=8, medium_size=12, bigger_size=14)

    max_amp = max_axis_value(x)

    fig1, ax1 = plt.subplots()
    fig1.suptitle('Actual Dataset versus Autoencoder Predicted Reconstruction')
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    ax1.plot(x[0, :], color='blue', label='Actual')
    ax1.plot(x_hat[0, :, 0], color='red', label='Predicted')
    ax1.set_ylim([-max_amp, max_amp])
    ax1.legend()
    plt.show()

    fig2, ax2 = plt.subplots()
    fig2.suptitle('Difference Between Actual and Predicted Dataset')
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    ax2.plot(x[0, :] - x_hat[0, :, 0], color='black', label='Difference')
    ax2.set_ylim([-max_amp, max_amp])
    plt.show()

    if save_path != '':
        fig1.savefig(save_path + '_actl_pred.png')
        fig2.savefig(save_path + '_diff_actl_pred.png')


# Custom keras callback to plot losses while training.
class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the objective function and learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(self.metrics[metric],
                        label=metric)
            if metric != 'lr':
                if logs['val_' + metric]:
                    axs[i].plot(self.metrics['val_' + metric],
                                label='val_' + metric)
            else:
                axs[i].plot(self.metrics[metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


# Main function, doesn't do anything.
def main():
    """
    Main function of dataPlotting, does nothing.
    """
    pass


# Main
if __name__ == "__main__":
    print("dataPlotting Executed when ran directly")
