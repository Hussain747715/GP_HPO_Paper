# Data Preprocessing for the Lookahead VSP Project. Includes data loading and custom ML generators as well as some
# pre-processing steps and custom loss functions.
# This is a part of Automating Hyperparameter Optimization in Geophysics with Optuna: A Comparative Study paper
# by Hussain Almarzooq & Umair Bin Waheed

# Import required libraries:
import numpy as np
from nptyping import NDArray
from sklearn.preprocessing import MinMaxScaler
from keras.utils.data_utils import Sequence
import keras.backend as K
import os
import dataGeneration as dg


# Normalize Data using min-max scaling.
def normalize_data(data: NDArray, min_data: float, max_data: float, min_range: float = 0, max_range: float = 1) -> \
        {NDArray, float}:
    """
    Applies data normalization on input data.
    Arguments:
        data (NDArray): Dataset where normalization will be applied.
        min_data (float): Min value to use in normalization.
        max_data (float): Max value to use in normalization.
        min_range (float): Min value to normalize data into.
        max_range (float): Max value to normalize data into.
    Returns:
        Returns the normalized data and the scaler.
    """
    # define min max scaler
    scaler = MinMaxScaler(feature_range=(min_range, max_range))

    # calculate fit for scaler
    scaler.fit(np.array([min_data, max_data]).reshape(-1, 1))

    # Apply scaler on data:
    normalized_data = scaler.transform(np.transpose(data))

    return np.transpose(normalized_data), scaler


# Simple function to create a directory if it doesn't exist.
def directory_create(dir_path: str):
    """
        If folder doesn't exist, then create it.
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


# Class responsible for creating data generators for ML applications which allows for generating data on the fly for
# ML applications.
class CustomDataGen(Sequence):
    """
        Custom data generator for machine learning applications. Generates data on the fly.
    """

    def __init__(self, batch_size: int = 1, n: int = 10, gen_type: str = 'AE', transform_type: str = 'NA',
                 noisy_date: bool = True, scaling_flag: bool = False):
        """
            Initializer for the dataset generator class.
            Arguments:
                batch_size (int): The number of datasets to use in each batch for training, default is 1.
                n (int): total number of datasets to generate for each epoch of training.
                gen_type (str): Type of generator to be used which depends on the application type, can be 'ML_VEL',
                'DeNoise'.
                noisy_date (bool): Whether to add noise to the generated data or not.
                scaling_flag (bool): Whether to apply scaling to the generated data or not.
        """
        self.batch_size = batch_size  # batch_size for training (number of datasets to use in each batch)
        self.n = n  # total number of datasets to generate for each epoch.
        self.gen_type = gen_type  # For autoencoders or machine learning (changes output type)
        self.scaling_flag = scaling_flag  # Whether to scale the data or not.
        self.noisy_date = noisy_date  # Whether to add noise the data or not.
        self.dgc = dg.DataGenerationClass(run_number=self.n, save_data_to_file=0)  # create generator class.

    def on_epoch_end(self):
        pass  # do nothing on epoch end as data is generated (no need to shuffle).

    # Method responsible for getting the generated data and providing it to the keras model.
    def __getitem__(self, index: int):
        # Generate one batch of data.
        self.dgc.run_data_generation(gen_num_start=index, gen_num_end=index + self.batch_size)

        # return x & y pairs for different ml applications.

        # default loaded x & y types.
        x_data_type = 'CONV_REFL_GAPS'
        y_data_type = 'VEL_GAPS'

        if self.gen_type == 'ML_VEL':  # Deep learning for velocity inversion. Returns convolved data (x) and velocity (y).
            if self.noisy_date == True:
                x_data_type = 'CONV_REFL_GAPS'
                y_data_type = 'VEL_GAPS'
            elif self.noisy_date == False:
                x_data_type = 'CONV_REFL_CLEAN'
                y_data_type = 'VEL'
        elif self.gen_type == 'DeNoise':  # Deep learning for denoising autoencoder applications. Returns noisy convolved
            # data (x) and clean convolved (y).
            x_data_type = 'CONV_REFL_NOISY'
            y_data_type = 'CONV_REFL_CLEAN'


        x, y = self.dgc.retrieve_mult_input_output(k_start=index, k_end=index + self.batch_size,
                                                           x_data=x_data_type, y_data=y_data_type)

        # scale the data if scaling flag is set to true.
        if self.scaling_flag == True:

            if x_data_type == 'REFL' or x_data_type == 'CONV_REFL_CLEAN' or x_data_type == 'CONV_REFL_NOISY' \
                    or x_data_type == 'CONV_REFL_GAPS':
                x, x_scaler = normalize_data(x, -1, 1, -1, 1)
            elif x_data_type == 'VEL' or x_data_type == 'VEL_GAPS':
                x, x_scaler = normalize_data(x, 0, 30000, 0, 1)

            if y_data_type == 'REFL' or y_data_type == 'CONV_REFL_CLEAN' or y_data_type == 'CONV_REFL_NOISY' \
                    or y_data_type == 'CONV_REFL_GAPS':
                y, y_scaler = normalize_data(y, -1, 1, -1, 1)
            elif y_data_type == 'VEL' or y_data_type == 'VEL_GAPS':
                y, y_scaler = normalize_data(y, 0, 30000, 0, 1)

        # Return appropriate pair for either AE or ML.
        if self.gen_type == 'AE':
            return x, x
        elif self.gen_type == 'ML_VEL':
            return x, y
        else:
            print(self.gen_type + 'is not a valid generator type, choose either ML_VEL or DeNoise, defaulting to ML_VEL.')
            return x, y

    def __len__(self):  # Required function for generators.
        return int(self.n // self.batch_size)


# Custom Loss Functions for keras:

# RMSE Loss.
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# LOG RMSE Loss.
def log_root_mean_squared_error(y_true, y_pred):
    return K.log(K.sqrt(K.mean(K.square((y_pred - y_true)))))

# if this file is run it loads one and plots one dataset transformation to CWT and SWT, to generate multiple datasets change run_number to the
# desired value, if one wants to save the files then should save_data_to_file=1
def main():
    """
    Main function of dataPreProcessing.
    """
    pass


# Main
if __name__ == "__main__":
    print("dataPreProcessing Executed when ran directly")
    main()
