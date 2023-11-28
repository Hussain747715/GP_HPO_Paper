# This file is responsible for all synthetic data generation in the Lookahead VSP Project.
# This is a part of Automating Hyperparameter Optimization in Geophysics with Optuna: A Comparative Study paper
# by Hussain Almarzooq & Umair Bin Waheed

# Import required libraries:
import numpy as np
import random
import os
import dataPlotting
from bruges.filters import wavelets
from nptyping import NDArray
from typing import Union, Iterable


#  Calculate wavelet then convolve with series
def wavelet_convolution(wavelet_type: str, data_to_conv: NDArray, dt_wavelet: float):
    """
    Convolve dataset with a wavelet and returns the convolved dataset.
    Arguments:
        wavelet_type (str): Determines the wavelet type. It can be either Ricker, Klauder, Orsmby, or All if all three
        types are desired.
        data_to_conv (NDArray): the dataset that needs to be convolved. Expects an NDArray.
        dt_wavelet (float): Sampling rate of wavelet. Should be the same as data.
    Returns:
        A single convolved dataset if either wavelet_type == Ricker, Klauder, or Orsmby.
        Otherwise, returns the data convolved with all three wavelets if wavelet_type == All.
    """

    # Set empty data  containers.
    convolved_data_ricker = None
    convolved_data_klauder = None
    convolved_data_orsmby = None
    convolved_data = None

    wavelet_duration = 0.512  # Wavelet length

    if wavelet_type == 'Ricker' or wavelet_type == 'All':  # Calculate Ricker wavelet
        ricker_freq = np.random.uniform(10.0, 50.0)
        ricker_wavelet, ricker_t = wavelets.ricker(wavelet_duration, dt_wavelet, ricker_freq, t=None, return_t=True,
                                                   sym=True)
        convolved_data_ricker = np.convolve(data_to_conv, ricker_wavelet, mode='same')
        convolved_data = convolved_data_ricker

    if wavelet_type == 'Klauder' or wavelet_type == 'All':  # Calculate Klauder wavelet
        klauder_freq = (np.random.uniform(2.0, 8.0), np.random.uniform(80.0, 120.0))
        klauder_wavelet, klauder_t = wavelets.klauder(wavelet_duration, dt_wavelet, klauder_freq,
                                                      autocorrelate=True,
                                                      t=None, return_t=True, taper='blackman', sym=True)
        convolved_data_klauder = np.convolve(data_to_conv, klauder_wavelet, mode='same')
        convolved_data = convolved_data_klauder

    if wavelet_type == 'Orsmby' or wavelet_type == 'All':  # Calculate Orsmby wavelet
        orsmby_freq = (np.random.uniform(2.0, 8.0), np.random.uniform(10.0, 20.0), np.random.uniform(60.0, 80.0),
                       np.random.uniform(90.0, 120.0))
        orsmby_wavelet, orsmby_t = wavelets.ormsby(wavelet_duration, dt_wavelet, orsmby_freq, t=None, return_t=True,
                                                   sym=True)
        convolved_data_orsmby = np.convolve(data_to_conv, orsmby_wavelet, mode='same')
        convolved_data = convolved_data_orsmby

    # Sanity check
    if wavelet_type != 'Ricker' and wavelet_type != 'Klauder' and wavelet_type != 'Orsmby' and wavelet_type != 'All':
        print(
            'This wavelet is not supported or unrecognized, the function only  supports Ricker, Klauder, or Orsmby '
            'wavelets')
        return None

    # Either return all wavelets convolutions ('All') or return the selected wavelet convolution only.
    if wavelet_type == 'All':  # Return all wavelets
        return convolved_data_ricker, convolved_data_klauder, convolved_data_orsmby
    else:  # Return one wavelet
        return convolved_data


# Add Noise to a Signal
def data_noise(data_to_noise: NDArray) -> NDArray:
    """
    Adds random noise to dataset.
    Arguments:
        data_to_noise (NDArray): the dataset that noise will be added to. Expects an NDArray.
    Returns:
        A dataset with the noise added.
    """

    # estimate gaussian noise parameters
    mu = float(np.mean(data_to_noise))
    sigma = float(np.std(data_to_noise))

    # Add noise to  the data.
    noisy_data = data_to_noise + [np.random.uniform(0.001, 1.0) * random.gauss(mu, sigma) for _ in
                                  range(len(data_to_noise))]
    return noisy_data


# Add gaps to signal
def data_gaps(data_to_gap: NDArray, top_mute: int = 0, bottom_mute: int = 0) -> {NDArray, int, int}:
    """
    Creates a random gap in the dataset.
    Arguments:
        data_to_gap (NDArray): the dataset that gap will be added to.
        top_mute (int): the upper limit of the mute.
        bottom_mute (int): the lower limit of the mute.
    Returns:
        A dataset with the gap added.
    """

    # Calculate top and bottom mutes.
    if top_mute == 0:
        top_mute = np.random.randint(0, len(data_to_gap) - 1)
        bottom_mute = np.random.randint(top_mute, len(data_to_gap) - 1)

    # Create data with the gap.
    data_with_gap = np.copy(data_to_gap)
    data_with_gap[top_mute:bottom_mute] = 0

    return data_with_gap, top_mute, bottom_mute


# Average velocity & density in gap section
def data_average(data_to_average: NDArray, top_mute: int, bottom_mute: int) -> NDArray:
    """
    Averages the dataset in the gap section of the dataset.
    Arguments:
        data_to_average (NDArray): the dataset that the gap will be averaged.
        top_mute (int): the upper limit of the mute.
        bottom_mute (int): the lower limit of the mute.
    Returns:
        A dataset with the gap averaged.
    """

    # Averages the velocity and density values in the gap section created in data_gaps function.
    averaged_data = np.copy(data_to_average)
    averaged_data[top_mute:bottom_mute] = np.mean(data_to_average[top_mute:bottom_mute])

    return averaged_data


# Save data to disk:
def save_data(save_folder_path: str, save_file_name: str, data_to_save: NDArray, precision: Union[str, Iterable, None],
              save_as_txt: int):
    """
    Saves dataset to the specified location as either txt or npy file.
    Arguments:
        save_folder_path (str): Folder path for the file to be saved.
        save_file_name (str): The file name for the file to be saved.
        data_to_save (NDArray): The data to be saved.
        precision: The desired precision of the dat to be saved.
        save_as_txt (int): Saves the dataset as either txt (1) or npy (0).
    Returns:
        No returns, saves files to desired location.
    """

    # Create folder if it doesn't exist.
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Save as either txt or npy file.
    if save_as_txt == 1:  # Invalid warning for data_to_save type.
        np.savetxt((save_folder_path + save_file_name + '.txt'), data_to_save, precision)
    else:
        np.save((save_folder_path + save_file_name), data_to_save)


# Class responsible for generating all data in this project.
class DataGenerationClass:
    """
    A class that deals with data generation for the lookahead project. It generates synthetic reflectivity series that
    can be convolved with certain wavelets, noise and gaps can be added. It can also return the velocity and density of
    the model that was used to create the datasets.
    """

    def __init__(self, variable_dt: int = 0, variable_data_length: int = 0, constant_data_length: int = 4096,
                 density_empirical: int = 1, convolve_with_wavelets: int = 1, add_noise: int = 1, add_gaps: int = 1,
                 save_data_to_file: int = 1, save_data_as_txt: int = 0, plotting_option: int = 0,
                 path_to_save: str = 'Data/', run_number: int = 1, t0: float = 0.0):
        """
            Initializer for the dataset generator class.
            Arguments:
                variable_dt (int): Set to either variable (1) or constant (0) sampling rate.
                variable_data_length (int): Set to either variable (1) or constant (0) data length .
                constant_data_length (int): If variable_data_length=0, sets the value for the constant data length.
                density_empirical (int): Whether to use random (0) or empirical (1) relationship for calculating density.
                convolve_with_wavelets (int): Whether to convolve reflectivity with wavelets (1) or keep it as is (0).
                add_noise (int): Whether to add gaussian noise to the data (1) or keep it as is (0).
                add_gaps (int): Whether to add gaps to the data (1) or keep it as is (0).
                save_data_to_file (int): Whether to save the data (1) or not (0).
                save_data_as_txt (int): Whether to save the data as a .txt file (1) or .npy file (0).
                plotting_option (int): Whether to plot (1) the generated model & convolved datasets or not (0).
                path_to_save (str): Path to save the generated data if save_data_to_file=1.
                run_number (int): Number of datasets to generate.
                t0 (float): Start time of the file, default is 0.0.
        """
        # Settings:
        self.variable_dt = variable_dt  # if set to 1 will allow variable dt in generated data (might not be suitable
        # for ML work).
        self.variable_data_length = variable_data_length  # if set to 1 will allow variable max data (might not be
        # suitable for ML work).
        self.constant_data_length = constant_data_length  # set constant number of samples if variable_data_length = 0
        self.density_empirical = density_empirical  # set to 1 if you want density to have some empirical
        # relationship to velocity, set to 0 if truly
        # random density is desired.
        self.convolve_with_wavelets = convolve_with_wavelets  # set to 1 if you want to convolve data with wavelets
        self.add_noise = add_noise  # set to 1 if you want to add noise to kw,rw,ow. Only works if
        # convolve_with_wavelets = 1.
        self.add_gaps = add_gaps  # set to 1 if you want to add gaps to kw,rw,ow.
        self.save_data_to_file = save_data_to_file  # choose whether to save the generated data or not
        self.save_data_as_txt = save_data_as_txt  # choose whether to use text format (1) or python format (0).
        self.plotting_option = plotting_option  # whether to plot generated plots or not
        self.path_to_save = path_to_save

        ################################################################################################################

        # Number of random data to create:
        self.run_number = run_number

        ################################################################################################################

        # Fixed Parameters:
        self.t0 = t0  # initial time in seconds

        ################################################################################################################

        # Datasets:
        # Determine whether dt is variable or constant
        if variable_dt == 1:  # Variable dt
            dt_possible = [1.0, 2.0, 4.0, 8.0]
        else:  # Constant dt
            dt_possible = [1.0]

        self.dt = 0.001 * np.random.choice(dt_possible)  # time step in seconds

        # Determine whether data length is constant or variable:
        if variable_data_length == 1:  # Variable data length
            self.nsamples = np.random.randint(400, 6000)
        else:  # Constant data length
            self.nsamples = constant_data_length

        # Calculated Parameters:
        self.lithology_type = None
        self.depth_layer = None
        self.vel_layer = None
        self.dens_layer = None
        self.imp_layer = None
        self.tmax = self.dt * self.nsamples  # end time in seconds
        self.t = np.arange(t0, self.tmax, self.dt)  # time series
        self.lith_time = np.zeros([self.run_number, len(self.t)])  # lithology profile in time
        self.vel_time = np.zeros([self.run_number, len(self.t)])  # velocity profile in time
        self.dens_time = np.zeros([self.run_number, len(self.t)])  # density profile in time
        self.imp_time = np.zeros([self.run_number, len(self.t)])  # imp profile in time
        self.refl_time = np.zeros([self.run_number, len(self.t)])  # reflectivity profile in time
        self.refl_time_ricker = np.zeros([self.run_number, len(self.t)])  # reflectivity ricker
        self.refl_time_klauder = np.zeros([self.run_number, len(self.t)])  # reflectivity klauder
        self.refl_time_orsmby = np.zeros([self.run_number, len(self.t)])  # reflectivity orsmby
        self.refl_time_ricker_noise = np.zeros([self.run_number, len(self.t)])  # reflectivity ricker noise
        self.refl_time_klauder_noise = np.zeros([self.run_number, len(self.t)])  # reflectivity klauder noise
        self.refl_time_orsmby_noise = np.zeros([self.run_number, len(self.t)])  # reflectivity orsmby noise
        self.refl_time_ricker_gaps = np.zeros([self.run_number, len(self.t)])  # reflectivity ricker noise + gaps
        self.refl_time_klauder_gaps = np.zeros([self.run_number, len(self.t)])  # reflectivity klauder noise + gaps
        self.refl_time_orsmby_gaps = np.zeros([self.run_number, len(self.t)])  # reflectivity orsmby noise + gaps
        self.vel_time_gap = np.zeros([self.run_number, len(self.t)])  # velocity profile in time + gap
        self.dens_time_gap = np.zeros([self.run_number, len(self.t)])  # density profile in time + gap
        self.imp_time_gap = np.zeros([self.run_number, len(self.t)])  # impedance profile in time + gap

        ################################################################################################################

    # Create velocity and density logs generation for all layers from lithology.
    def velocity_density_logs_creation(self, lithology_type: list[int], depth_layer: list[int], nlayers: int) -> \
            {list[int], list[int]}:
        """
        Velocity and density log generation.
        Arguments:
            lithology_type (list[int]): List of lithologies to be created.
            depth_layer (list[int]): List of layer depths to be created.
            nlayers (int): Number of layers to be created.
        Returns:
            Two lists one containing layer velocity and the other containing the density.
        """

        # Lithology parameters based on Mavko et al., 2020, equation p = a*v^2 + b*v + c:
        litho_var_a = [-0.0261, -0.0115, -0.0296, -0.0235, -0.0203, -0.0455]
        litho_var_b = [0.373, 0.261, 0.461, 0.39, 0.321, 0.5892]
        litho_var_c = [1.458, 1.515, 0.963, 1.242, 1.732, 0.4358]

        # Lower and upper limits of velocity for each lithology
        litho_vel_low_km = [1.5, 1.5, 3.5, 4.5, 4.6, 4.37]
        litho_vel_high_km = [5, 6, 6.4, 7.1, 7.4, 4.8]

        # Compaction trend control for each lithology.
        depth_control_lower = [1.5, 1.5, 1.5, 1.5, 1.5, 1]
        depth_control_upper = [-2.0, -2.0, -1.8, -1.8, -1.5, -1]

        # Create containers for average velocity and density of each layer:
        vel_layer = np.zeros(nlayers)
        dens_layer = np.zeros(nlayers)

        # Loop to generate density and velocity based on current lithology
        for i in range(0, nlayers):
            # Current lithology
            litho_curr = lithology_type[i]

            # Layer interval velocities km/s
            depth_factor_low = np.interp(depth_layer[i], (0, 10000), (1, depth_control_lower[litho_curr]))
            depth_factor_high = np.abs(np.interp(depth_layer[i], (0, 10000), (depth_control_upper[litho_curr], -1)))
            vel_layer[i] = np.random.uniform(litho_vel_low_km[litho_curr] * depth_factor_low,
                                             litho_vel_high_km[litho_curr] / depth_factor_high, 1)

            # Determine whether density is random or empirical and then calculate the density.
            if self.density_empirical == 1:  # empirical density
                dens_layer[i] = (litho_var_a[litho_curr] * np.square(vel_layer[i])) + (
                        litho_var_b[litho_curr] * vel_layer[i]) + litho_var_c[litho_curr]  # density values g/cm^3
            else:  # random density
                dens_layer[i] = (0.1 * np.random.rand(nlayers)) + (
                    np.random.uniform(1.9, 3.1, nlayers))  # density values g/cm^3

        # Convert velocity to feet:
        vel_layer = vel_layer * 3.28084 * 1000

        return vel_layer, dens_layer

    # Function to create velocity, density, and reflectivity data
    def create_data(self, k: int = 0) -> {NDArray, float, NDArray, NDArray, NDArray}:
        """
        Creates velocity, density, and reflectivity data
        Arguments:
            k (int): Current dataset number.
        Returns:
            Time, sampling rate, reflectivity, velocity, and density.
        """

        # Load init. parameters:
        seq_lith_cur = None
        lith_time = self.lith_time[k, :]  # lithology profile in time
        vel_time = self.vel_time[k, :]  # velocity profile in time
        dens_time = self.dens_time[k, :]  # density profile in time
        refl_time = self.refl_time[k, :]  # reflectivity profile in time

        # Randomize layer parameters:

        # First create main sequences (clastics, carbonates, or salts):
        nsequences = np.random.randint(1, 100)  # number of sequences
        sequence_type = np.random.choice(3, nsequences, p=[0.65, 0.30, 0.05])  # 0) Shale and Sand, 1) Shale, Limestone,
        # Dolomite, Anhydrite; 3) halite and shale and anhydrite.
        seq_0 = [0, 1]  # Clastics contains only shale and sand lithologies
        seq_1 = [0, 2, 3, 4]  # Carbonates contains shale, limestone, dolmite, and anhydrite
        seq_2 = [0, 4, 5]  # Salts contains shale, halite and anhydrite.

        # Loop over to fill each sequence with layers
        nlayers = -1  # initial number of layers
        lithology_type = [0]
        thickness_layer = [0]
        depth_layer_current = 0
        for i in range(0, nsequences):

            # Create layers within sequences:
            seq_layer_cur = np.random.randint(3, 10)

            if sequence_type[i] == 0:
                seq_lith_cur = np.random.choice(seq_0, seq_layer_cur)
            elif sequence_type[i] == 1:
                seq_lith_cur = np.random.choice(seq_1, seq_layer_cur, p=[0.30, 0.30, 0.30, 0.10])
            elif sequence_type[i] == 2:
                seq_lith_cur = np.random.choice(seq_2, seq_layer_cur, p=[0.40, 0.40, 0.20])

            # Calculate thickness & depth for stopping condition
            thickness_layer_current = np.random.uniform(10.0, 1000.0, seq_layer_cur)
            depth_layer_current += np.sum(thickness_layer_current)
            thickness_layer = np.concatenate((thickness_layer, thickness_layer_current), axis=0)
            nlayers += seq_layer_cur

            # Create array of lithologies for all sequences:
            # 0) Shale, 1) Sandstone, 2) Limestone, 3) Dolomite, 4) Anhydrite, 5) Salt.
            if i == 0:
                lithology_type = seq_lith_cur
            else:
                lithology_type = np.concatenate((lithology_type, seq_lith_cur), axis=0)

            # Stopping condition at random between 10k and 25k depth.
            if depth_layer_current >= np.random.uniform(10000.0, 25000.0):
                break

        lithology_type = lithology_type[1:]
        thickness_layer = thickness_layer[2:]  # thickness of layers in ft

        # Calculating the rest of the parameters:
        depth_layer = np.cumsum(thickness_layer)  # depth to base

        # Calculate velocity and density of each layer.
        vel_layer, dens_layer = self.velocity_density_logs_creation(lithology_type, depth_layer, nlayers)

        imp_layer = vel_layer * dens_layer  # acoustic impedance

        # Reflection coefficients for the layers
        refc_layer = ((imp_layer[2:-1] - imp_layer[1:-2]) / (imp_layer[2:-1] + imp_layer[1:-2]))
        np.insert(refc_layer, 0, 0.0)
        np.append(refc_layer, 0.0)

        # Injecting refc_layer into time series & calculating velocities & densities in time:
        time_layer = np.zeros(len(refc_layer))
        refc_time_index = np.zeros(len(refc_layer))

        # Sanity check
        if time_layer.size == 0:
            time_layer = np.zeros(1)
        if thickness_layer.size == 0:
            thickness_layer = np.zeros(1)
        if refc_time_index.size == 0:
            refc_time_index = np.zeros(1)

        # Calculating properties in time for first layer:
        time_layer[0] = (2.0 * thickness_layer[0] / vel_layer[0])
        refc_time_index[0] = np.argmin(abs(self.t - time_layer[0]))
        vel_time[0:int(refc_time_index[0])] = vel_layer[0]
        dens_time[0:int(refc_time_index[0])] = dens_layer[0]
        lith_time[0:int(refc_time_index[0])] = lithology_type[0]

        # Calculating properties in time for the remaining layers:
        for i in range(1, len(time_layer) - 1):
            time_layer[i] = time_layer[i - 1] + (2.0 * thickness_layer[i] / vel_layer[i])  # time to layers in s

            if self.tmax > time_layer[i]:
                refc_time_index[i] = np.argmin(abs(self.t - time_layer[i]))

                # Calculating velocities & densities in time
                vel_time[int(refc_time_index[i - 1]):int(refc_time_index[i])] = vel_layer[i]
                dens_time[int(refc_time_index[i - 1]):int(refc_time_index[i])] = dens_layer[i]
                lith_time[int(refc_time_index[i - 1]):int(refc_time_index[i])] = lithology_type[i]

            elif int(refc_time_index[i - 1]) > 0:
                if i == len(time_layer) - 1:  # Filling last layer till end
                    vel_time[int(refc_time_index[i - 1]):-1] = vel_layer[i]
                    dens_time[int(refc_time_index[i - 1]):-1] = dens_layer[i]
                    lith_time[int(refc_time_index[i - 1]):-1] = lithology_type[i]
                    break

        vel_time[vel_time == 0] = vel_layer[-1]
        dens_time[dens_time == 0] = dens_layer[-1]
        lith_time[lith_time == 0] = lithology_type[-1]

        # Final Outputs (adding small variations to stimulate intra-layer variations):
        vel_time = vel_time + np.random.uniform(-500.0, 500., len(vel_time))
        dens_time = dens_time + (np.random.uniform(-0.09, 0.09, len(dens_time)))

        imp_time = vel_time * dens_time
        refl_time[1:-2] = ((imp_time[2:-1] - imp_time[1:-2]) / (imp_time[2:-1] + imp_time[1:-2]))

        # Update datasets in self:

        # Average properties per layer.
        self.lithology_type = lithology_type
        self.depth_layer = depth_layer
        self.vel_layer = vel_layer
        self.dens_layer = dens_layer
        self.imp_layer = imp_layer

        # Properties in time.
        self.lith_time[k, :] = lith_time  # lithology profile in time
        self.vel_time[k, :] = vel_time  # velocity profile in time
        self.dens_time[k, :] = dens_time  # density profile in time
        self.imp_time[k, :] = imp_time  # imp profile in time
        self.refl_time[k, :] = refl_time  # reflectivity profile in time
        return self.t, self.dt, vel_time, dens_time, refl_time

    # Generates 1 dataset
    def generate_dataset(self, k: int = 0):
        """
        Generates a single complete dataset.
        Arguments:
            k (int): number of dataset (used for saving and indexing purposes).
        Returns:
            None.
        """
        # Calculate a single model and its synthetic data:
        t_current, dt_current, vel_time_current, dens_time_current, refl_time_current = self.create_data(k)
        imp_time_current = vel_time_current * dens_time_current

        # Convolve data with all three wavelets (ricker, klauder, orsmby):
        refl_time_current_ricker, refl_time_current_klauder, refl_time_current_orsmby = wavelet_convolution(
            'All', refl_time_current, dt_current)

        # Add noise to the data if option is selected:
        if self.add_noise == 1:
            refl_time_current_ricker_noise = data_noise(refl_time_current_ricker)
            refl_time_current_klauder_noise = data_noise(refl_time_current_klauder)
            refl_time_current_orsmby_noise = data_noise(refl_time_current_orsmby)
        else:
            refl_time_current_ricker_noise = refl_time_current_ricker
            refl_time_current_klauder_noise = refl_time_current_klauder
            refl_time_current_orsmby_noise = refl_time_current_orsmby

        # Add gaps to the data if option is selected:
        if self.add_gaps == 1:
            refl_time_current_ricker_gaps, top_mute_current, bottom_mute_current = data_gaps(
                refl_time_current_ricker_noise)
            refl_time_current_klauder_gaps, top_mute_current, bottom_mute_current = data_gaps(
                refl_time_current_klauder_noise,
                top_mute_current,
                bottom_mute_current)
            refl_time_current_orsmby_gaps, top_mute_current, bottom_mute_current = data_gaps(
                refl_time_current_orsmby_noise,
                top_mute_current,
                bottom_mute_current)
            vel_time_current_gap = data_average(vel_time_current, top_mute_current, bottom_mute_current)
            dens_time_current_gap = data_average(dens_time_current, top_mute_current, bottom_mute_current)
            imp_time_current_gap = data_average(imp_time_current, top_mute_current, bottom_mute_current)
        else:
            refl_time_current_ricker_gaps = refl_time_current_ricker_noise
            refl_time_current_klauder_gaps = refl_time_current_klauder_noise
            refl_time_current_orsmby_gaps = refl_time_current_orsmby_noise
            vel_time_current_gap = vel_time_current
            dens_time_current_gap = dens_time_current
            imp_time_current_gap = imp_time_current

        # Update datasets in self:
        self.refl_time_ricker[k, :] = refl_time_current_ricker  # reflectivity ricker
        self.refl_time_klauder[k, :] = refl_time_current_klauder  # reflectivity klauder
        self.refl_time_orsmby[k, :] = refl_time_current_orsmby  # reflectivity orsmby
        self.refl_time_ricker_noise[k, :] = refl_time_current_ricker_noise  # reflectivity ricker noise
        self.refl_time_klauder_noise[k, :] = refl_time_current_klauder_noise  # reflectivity klauder noise
        self.refl_time_orsmby_noise[k, :] = refl_time_current_orsmby_noise  # reflectivity orsmby noise
        self.refl_time_ricker_gaps[k, :] = refl_time_current_ricker_gaps  # reflectivity ricker noise + gaps
        self.refl_time_klauder_gaps[k, :] = refl_time_current_klauder_gaps  # reflectivity klauder noise + gaps
        self.refl_time_orsmby_gaps[k, :] = refl_time_current_orsmby_gaps  # reflectivity orsmby noise + gaps
        self.vel_time_gap[k, :] = vel_time_current_gap  # velocity profile in time + gap
        self.dens_time_gap[k, :] = dens_time_current_gap  # density profile in time + gap
        self.imp_time_gap[k, :] = imp_time_current_gap  # density profile in time + gap

        # Save dataset if the option is selected.
        if self.save_data_to_file == 1:
            self.save_generated_data(k)

        # Plotting if the option is selected.
        if self.plotting_option == 1:
            self.plot_generated_data(k)

    # Save generated data
    def save_generated_data(self, k: int = 0):
        """
        Saves all generated data.
        Arguments:
            k (int): number of dataset (used for saving and indexing purposes).
        Returns:
            None.
        """

        # Path to save and option to save as .txt or .npy
        path_to_save = self.path_to_save
        save_data_as_txt = self.save_data_as_txt

        # save time if variable:
        if self.variable_data_length == 1:
            save_data((path_to_save + 'TIME/'), ('TIME_' + str(k).zfill(10)), self.t,
                      ['%10.32f'], save_data_as_txt)
        elif self.variable_data_length == 0 and k == 0:
            save_data((path_to_save + 'TIME/'), 'TIME_ALL', self.t,
                      ['%10.32f'], save_data_as_txt)

        # save dt if variable:
        if self.variable_dt == 1:
            save_data((path_to_save + 'DT/'), ('DT_' + str(k).zfill(10)),
                      np.atleast_1d(self.dt), ['%2.0f'], save_data_as_txt)
        elif self.variable_dt == 0 and k == 0:
            save_data((path_to_save + 'DT/'), 'DT_ALL',
                      np.atleast_1d(self.dt), ['%2.0f'], save_data_as_txt)

        # Save clean data
        save_data((path_to_save + 'VEL/'), ('VEL_' + str(k).zfill(10)), self.vel_time[k, :],
                  ['%10.32f'], save_data_as_txt)
        save_data((path_to_save + 'DENS/'), ('DENS_' + str(k).zfill(10)), self.dens_time[k, :],
                  ['%10.32f'], save_data_as_txt)
        save_data((path_to_save + 'REFL/'), ('REFL_' + str(k).zfill(10)), self.refl_time[k, :],
                  ['%10.32f'], save_data_as_txt)
        save_data((path_to_save + 'REFL_RW/'), ('REFL_RW_' + str(k).zfill(10)),
                  self.refl_time_ricker[k, :], ['%10.32f'], save_data_as_txt)
        save_data((path_to_save + 'REFL_KW/'), ('REFL_KW_' + str(k).zfill(10)),
                  self.refl_time_klauder[k, :], ['%10.32f'], save_data_as_txt)
        save_data((path_to_save + 'REFL_OW/'), ('REFL_OW_' + str(k).zfill(10)),
                  self.refl_time_orsmby[k, :], ['%10.32f'], save_data_as_txt)

        # save noisy data if generated:`
        if self.add_noise == 1:
            save_data((path_to_save + 'REFL_RW_NOISE/'), ('REFL_RW_NOISE_' + str(k).zfill(10)),
                      self.refl_time_ricker_noise[k, :], ['%10.32f'], save_data_as_txt)
            save_data((path_to_save + 'REFL_KW_NOISE/'), ('REFL_KW_NOISE_' + str(k).zfill(10)),
                      self.refl_time_klauder_noise[k, :], ['%10.32f'], save_data_as_txt)
            save_data((path_to_save + 'REFL_OW_NOISE/'), ('REFL_OW_NOISE_' + str(k).zfill(10)),
                      self.refl_time_orsmby_noise[k, :], ['%10.32f'], save_data_as_txt)

        # save data with gaps if generated:`
        if self.add_gaps == 1:
            save_data((path_to_save + 'VEL_GAPS/'), ('VEL_GAPS_' + str(k).zfill(10)),
                      self.vel_time_gap[k, :], ['%10.32f'], save_data_as_txt)
            save_data((path_to_save + 'DENS_GAPS/'), ('DENS_GAPS_' + str(k).zfill(10)),
                      self.dens_time_gap[k, :], ['%10.32f'], save_data_as_txt)
            save_data((path_to_save + 'REFL_RW_GAPS/'), ('REFL_RW_GAPS_' + str(k).zfill(10)),
                      self.refl_time_ricker_gaps[k, :], ['%10.32f'], save_data_as_txt)
            save_data((path_to_save + 'REFL_KW_GAPS/'), ('REFL_KW_GAPS_' + str(k).zfill(10)),
                      self.refl_time_klauder_gaps[k, :], ['%10.32f'], save_data_as_txt)
            save_data((path_to_save + 'REFL_OW_GAPS/'), ('REFL_OW_GAPS_' + str(k).zfill(10)),
                      self.refl_time_orsmby_gaps[k, :], ['%10.32f'], save_data_as_txt)

    # Plot generated data
    def plot_generated_data(self, k: int = 0):
        """
        Creates several plots of generated data.
        """
        # Plots model (lithology, velocity, density, and impedance) in depth.
        dataPlotting.plot_model(self.lithology_type, self.depth_layer, self.vel_layer,
                                self.dens_layer, self.imp_layer)

        # Crossplot of velocity and density colored by lithology.
        dataPlotting.plot_xplot(self.lith_time[k, :], self.vel_time[k, :], self.dens_time[k, :])

        # Crossplot of velocity and density colored by impedance.
        dataPlotting.plot_xplot(self.lith_time[k, :], self.vel_time[k, :], self.dens_time[k, :], self.imp_time[k, :])

        # Plots of data from reflecitivty series to convolved data for all three wavelets as well as a comparison
        # between all three
        dataPlotting.plot_refl(self.t, self.refl_time[k, :], self.refl_time_ricker[k, :],
                               self.refl_time_ricker_noise[k, :], self.refl_time_ricker_gaps[k, :], "Ricker")
        dataPlotting.plot_refl(self.t, self.refl_time[k, :], self.refl_time_klauder[k, :],
                               self.refl_time_klauder_noise[k, :], self.refl_time_klauder_gaps[k, :], "Klauder")
        dataPlotting.plot_refl(self.t, self.refl_time[k, :], self.refl_time_orsmby[k, :],
                               self.refl_time_orsmby_noise[k, :], self.refl_time_orsmby_gaps[k, :], "Orsmby")
        dataPlotting.plot_refl_compare(self.t, self.refl_time[k, :], self.refl_time_ricker[k, :],
                                       self.refl_time_klauder[k, :], self.refl_time_orsmby[k, :])

    # Retrieve One Velocity Reflectivity Pair:
    def retrieve_input_output(self, k: int = 0, wavelet_choice: int = -1, x_data: str = '',
                              y_data: str = '') -> {NDArray, NDArray}:
        """
        Retrieves a single complete dataset.
        Arguments:
            k (int): number of dataset (used for saving and indexing purposes).
            wavelet_choice (int): whether to use ricker (0), klauder (1) or orsmby (2) wavelet.
            x_data (str), y_data (str): Any combination of the following strings: REFL, CONV_REFL_CLEAN,
            CONV_REFL_NOISY, CONV_REFL_GAPS, VEL, VEL_GAPS
        Returns:
            Returns a single dataset for any selection of inputs and outputs.
        """
        if wavelet_choice == -1:
            wavelet_choice = np.random.choice(3, 1)

        x = self.refl_time[k, :]
        y = self.vel_time[k, :]

        # Ricker datasets:
        # Selects appropriate property or data to return for x_data (input).
        if wavelet_choice == 0:
            if x_data == 'REFL':
                x = self.refl_time[k, :]
            elif x_data == 'CONV_REFL_CLEAN':
                x = self.refl_time_ricker[k, :]
            elif x_data == 'CONV_REFL_NOISY':
                x = self.refl_time_ricker_noise[k, :]
            elif x_data == 'CONV_REFL_GAPS':
                x = self.refl_time_ricker_gaps[k, :]
            elif x_data == 'VEL':
                x = self.vel_time[k, :]
            elif x_data == 'VEL_GAPS':
                x = self.vel_time_gap[k, :]
            elif x_data == 'IMP':
                x = self.imp_time[k, :]
            elif x_data == 'IMP_GAPS':
                x = self.imp_time_gap[k, :]
            elif x_data == 'DENS':
                x = self.dens_time[k, :]
            elif x_data == 'DENS_GAPS':
                x = self.dens_time_gap[k, :]

            # Selects appropriate property or data to return for y_data (output).
            if y_data == 'REFL':
                y = self.refl_time[k, :]
            elif y_data == 'CONV_REFL_CLEAN':
                y = self.refl_time_ricker[k, :]
            elif y_data == 'CONV_REFL_NOISY':
                x = self.refl_time_ricker_noise[k, :]
            elif y_data == 'CONV_REFL_GAPS':
                y = self.refl_time_ricker_gaps[k, :]
            elif y_data == 'VEL':
                y = self.vel_time[k, :]
            elif y_data == 'VEL_GAPS':
                y = self.vel_time_gap[k, :]
            elif y_data == 'IMP':
                y = self.imp_time[k, :]
            elif y_data == 'IMP_GAPS':
                y = self.imp_time_gap[k, :]
            elif y_data == 'DENS':
                y = self.dens_time[k, :]
            elif y_data == 'DENS_GAPS':
                y = self.dens_time_gap[k, :]

        # Klauder datasets:
        # Selects appropriate property or data to return for x_data (input).
        elif wavelet_choice == 1:
            if x_data == 'REFL':
                x = self.refl_time[k, :]
            elif x_data == 'CONV_REFL_CLEAN':
                x = self.refl_time_klauder[k, :]
            elif x_data == 'CONV_REFL_NOISY':
                x = self.refl_time_klauder_noise[k, :]
            elif x_data == 'CONV_REFL_GAPS':
                x = self.refl_time_klauder_gaps[k, :]
            elif x_data == 'VEL':
                x = self.vel_time[k, :]
            elif x_data == 'VEL_GAPS':
                x = self.vel_time_gap[k, :]
            elif x_data == 'IMP':
                x = self.imp_time[k, :]
            elif x_data == 'IMP_GAPS':
                x = self.imp_time_gap[k, :]
            elif x_data == 'DENS':
                x = self.dens_time[k, :]
            elif x_data == 'DENS_GAPS':
                x = self.dens_time_gap[k, :]

            # Selects appropriate property or data to return for y_data (output).
            if y_data == 'REFL':
                y = self.refl_time[k, :]
            elif y_data == 'CONV_REFL_CLEAN':
                y = self.refl_time_klauder[k, :]
            elif y_data == 'CONV_REFL_NOISY':
                x = self.refl_time_klauder_noise[k, :]
            elif y_data == 'CONV_REFL_GAPS':
                y = self.refl_time_klauder_gaps[k, :]
            elif y_data == 'VEL':
                y = self.vel_time[k, :]
            elif y_data == 'VEL_GAPS':
                y = self.vel_time_gap[k, :]
            elif y_data == 'IMP':
                y = self.imp_time[k, :]
            elif y_data == 'IMP_GAPS':
                y = self.imp_time_gap[k, :]
            elif y_data == 'DENS':
                y = self.dens_time[k, :]
            elif y_data == 'DENS_GAPS':
                y = self.dens_time_gap[k, :]

        # Orsmby datasets:
        # Selects appropriate property or data to return for x_data (input).
        elif wavelet_choice == 2:
            if x_data == 'REFL':
                x = self.refl_time[k, :]
            elif x_data == 'CONV_REFL_CLEAN':
                x = self.refl_time_orsmby[k, :]
            elif x_data == 'CONV_REFL_NOISY':
                x = self.refl_time_orsmby_noise[k, :]
            elif x_data == 'CONV_REFL_GAPS':
                x = self.refl_time_orsmby_gaps[k, :]
            elif x_data == 'VEL':
                x = self.vel_time[k, :]
            elif x_data == 'VEL_GAPS':
                x = self.vel_time_gap[k, :]
            elif x_data == 'IMP':
                x = self.imp_time[k, :]
            elif x_data == 'IMP_GAPS':
                x = self.imp_time_gap[k, :]
            elif x_data == 'DENS':
                x = self.dens_time[k, :]
            elif x_data == 'DENS_GAPS':
                x = self.dens_time_gap[k, :]

            # Selects appropriate property or data to return for y_data (output).
            if y_data == 'REFL':
                y = self.refl_time[k, :]
            elif y_data == 'CONV_REFL_CLEAN':
                y = self.refl_time_orsmby[k, :]
            elif y_data == 'CONV_REFL_NOISY':
                x = self.refl_time_orsmby_noise[k, :]
            elif y_data == 'CONV_REFL_GAPS':
                y = self.refl_time_orsmby_gaps[k, :]
            elif y_data == 'VEL':
                y = self.vel_time[k, :]
            elif y_data == 'VEL_GAPS':
                y = self.vel_time_gap[k, :]
            elif y_data == 'IMP':
                y = self.imp_time[k, :]
            elif y_data == 'IMP_GAPS':
                y = self.imp_time_gap[k, :]
            elif y_data == 'DENS':
                y = self.dens_time[k, :]
            elif y_data == 'DENS_GAPS':
                y = self.dens_time_gap[k, :]

        return x, y

    # Retrieve Multiple Velocity Reflectivity Pair:
    def retrieve_mult_input_output(self, k_start: int = 0, k_end: int = 0, wavelet_choice: int = -1
                                   , x_data: str = '', y_data: str = ''):
        """
        Retrieves a k number of complete datasets containing reflecivity series (or convolved data) and velocity pairs.
        Arguments:
            k_start (int),  k_end (int): range of datasets to retrieve.
            wavelet_choice (int): whether to use ricker (0), klauder (1) or orsmby (2) wavelet.
            x_data (str), y_data (str): Any combination of the following strings: REFL, CONV_REFL_CLEAN,
            CONV_REFL_NOISY, CONV_REFL_GAPS, VEL, VEL_GAPS
        Returns:
            Returns a multiple datasets for any selection of inputs and outputs.
        """
        if k_end == 0:
            k_end = self.run_number

        x_dataset = np.zeros(np.shape(self.refl_time[k_start:k_end, :]))
        y_dataset = np.zeros(np.shape(self.vel_time_gap[k_start:k_end, :]))

        data_counter = 0
        for i in range(k_start, k_end):
            x, y = self.retrieve_input_output(i, wavelet_choice, x_data, y_data)
            x_dataset[data_counter, :] = x
            y_dataset[data_counter, :] = y
            data_counter += 1

        return x_dataset, y_dataset

    # Main Program generates an x number of datasets.
    def run_data_generation(self, gen_num_start: int = 0, gen_num_end: int = 0):
        """
        Generates multiple datasets based on run_number.
        """
        if gen_num_end == 0:
            gen_num_end = self.run_number

        for k in range(gen_num_start, gen_num_end):
            self.generate_dataset(k)


# if this file is run it generates and plots one dataset, to generate multiple datasets change run_number to the
# desired value, if one wants to save the files then should save_data_to_file=1
def main():
    """
    Main function of dataGeneration.
    """
    run_number = 1
    dgc = DataGenerationClass(run_number=run_number, plotting_option=1, save_data_to_file=0)
    dgc.run_data_generation()


# Main
if __name__ == "__main__":
    print("dataGeneration Executed when ran directly")
    main()
