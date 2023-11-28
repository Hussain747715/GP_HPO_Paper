# Data Base ML models for the Lookahead VSP Project. It includes a class that contains functions that can dynamically
# create, train, and test Keras ML models based on input hyperparameters and input types.
# This is a part of Automating Hyperparameter Optimization in Geophysics with Optuna: A Comparative Study paper
# by Hussain Almarzooq & Umair Bin Waheed

# Import required libraries:
import os
import datetime
import numpy as np
from nptyping import NDArray
import keras
import tensorflow as tf
from keras import layers, callbacks, optimizers, regularizers, Sequential
from keras.utils.data_utils import Sequence
from keras.utils.vis_utils import plot_model
from tcn import TCN
import dataGeneration as dg
import dataPreProcessing as dpp
import dataPlotting as dp
import csv


# The primary class that contains all functions related to Keras ML models.
class BaseMLClass:
    """
    A class that deals with data machine learning models for the lookahead project.
    """

    def __init__(self, trainingDataGen: Sequence, validationDataGen: Sequence, num_ml_layers: int = 2,
                 dnn_units: int = 1000, lstm_units: int = 10, num_filters: int = 64, kernel_size: int = 3,
                 max_dilation: int = 32, pooling_value: int = 1, dropout_rate: float = 0, activation_func: str = 'swish'
                 , optimizer_func: optimizers = optimizers.Adam(learning_rate=0.003, amsgrad=True),
                 regularizer_func: regularizers = regularizers.L1L2(l1=1e-5, l2=1e-4), monitor_metric: str = 'val_loss',
                 loss_func='MSE', num_epochs: int = 10, num_workers: int = 1, verbose_val: int = 1,
                 cbs: callbacks = None, use_advanced_lr: bool = True, load_custom_objects=None, disp_loss: bool = True,
                 model_name: str = 'Model_', path_to_save: str = 'Base_ML_Model\\'):
        """
        Initializer for the dataset generator class.
        Arguments:
            trainingDataGen (generator): Data generator for training.
            validationDataGen (generator): Data generator for validation.
            num_ml_layers (int): How many layers in encoder/decorder (ons side).
            num_filters (int): Number of filters in Conv1D/TCN layers.
            dnn_units (int): number of neurons in a dense layer.
            lstm_units (int): number of neurons in a LSTM layer.
            kernel_size (int): Kernel size for Conv1D/TCN layers.
            max_dilation (int): Maximum dilation number.
            activation_func (str): Activation function for Conv1D layers.
            pooling_value (int): Determine value of MaxPooling1D and UpSampling1D.
            dropout_rate (float): Determine rate of dropout in dropout layer.
            optimizer_func (keras.optimizer): Optimizer function for the CNN network.
            monitor_metric (str): Monitor metric.
            loss_func (str): Loss function for the ML network.
            num_epochs (int): Epoch number for training/validation.
            num_workers (int): number of workers for training if set to 1 swish to not use multiprocessing.
            verbose_val (int): Whether to be verbose (1) or not (0).
            cbs (callbacks): For non-default callbacks.
            use_advanced_lr (bool): Whether to use dwell callback.
            load_custom_objects: Loads custom objects for loaded models.
            disp_loss (bool): whether to display loss or not.
            model_name (str): Name of model used for saving.
            path_to_save (str): Path to save all model related files to.
        Returns:
            None.
        """
        if cbs is None:
            cbs = []
        self.trainingDataGen = trainingDataGen  # Generator used for training data
        self.validationDataGen = validationDataGen  # Generator used for validation data
        self.num_ml_layers = num_ml_layers  # Number of  layers to use for ML (both CNN/TCN).
        self.dnn_units = dnn_units  # number of dnn units in dense layer.
        self.lstm_units = lstm_units  # number of lstm units in LSTM layer.
        self.num_filters = num_filters  # Number of filters.
        self.kernel_size = kernel_size  # Kernel size.
        self.max_dilation = max_dilation  # Max dilation on which the list of dilations will be created.
        self.activation_func = activation_func  # Type of activation to be used for Conv1D and TCN layers.
        self.pooling_value = pooling_value  # Pooling value to use for MaxPooling1D and UpSampling1D.
        self.dropout_rate = dropout_rate  # Dropout rate
        self.optimizer_func = optimizer_func  # Type of optimizer function to use.
        self.regularizer_func = regularizer_func  # Type of regularizer to use.
        self.monitor_metric = monitor_metric  # Monitor metric to use for callbacks.
        self.loss_func = loss_func  # Type of loss function to use.
        self.num_epochs = num_epochs  # Maximum number of epochs.
        self.num_workers = num_workers  # Number of workers to use (1==single threaded).
        self.verbose_val = verbose_val  # Whether to be verbose while running the code.
        self.use_advanced_lr = use_advanced_lr  # Whether to use dwell and CLR callbacks or not.
        self.load_custom_objects = load_custom_objects  # Load custom objects for laoded models.
        self.disp_loss = disp_loss  # Whether to show the loss plots or not.
        self.model_name = model_name  # Name of the  model to be used while saving files.
        self.path_to_save = path_to_save  # Base path for model.
        self.ml_model = None  # Container for  the ml_model model.
        self.encoder = None  # Container for the encoder model.
        self.denoiser = None  # Container for the denoiser model.
        self.inversion = None  # Container for the inversion model.
        self.ml_model_type: str = ''  # Used for saving purposes to label as CNN/TCN.
        self.transform_type = trainingDataGen.transform_type  # Used for saving purposes to label input data based
        # on transformation if any.
        self.currentDate = datetime.datetime.now().strftime("%Y%m%d_%H%M")  # Date to use in saving the files.

        # Directories:
        self.file_name = self.path_to_save + self.model_name + self.transform_type + '_' + self.currentDate + '_best_model.h5'
        self.backup_dir = self.path_to_save + 'Backup\\' + self.model_name + self.transform_type + '_' + self.currentDate + '_backup_model '
        self.logs_dir = self.path_to_save + 'logs\\'
        self.loss_dir = self.path_to_save + 'history\\' + self.model_name + self.transform_type + '_' + self.currentDate + '_log.csv'

        # Callbacks for ML.
        self.es = callbacks.EarlyStopping(monitor=self.monitor_metric, mode='min', verbose=self.verbose_val,
                                          patience=100,
                                          min_delta=0)
        self.mc = callbacks.ModelCheckpoint(filepath=self.file_name, monitor=self.monitor_metric,
                                            verbose=self.verbose_val,
                                            mode='min', save_best_only=True)
        self.b_r = callbacks.BackupAndRestore(backup_dir=os.path.abspath(self.backup_dir))
        # tensorboard - -logdir = self.logs_dir to access tensorboard.
        self.tbg = callbacks.TensorBoard(log_dir=self.logs_dir, histogram_freq=1, write_images=True)
        self.red_lr_small = callbacks.ReduceLROnPlateau(monitor=self.monitor_metric, factor=0.95, patience=10,
                                                        cooldown=5, verbose=verbose_val)
        self.red_lr_big = callbacks.ReduceLROnPlateau(monitor=self.monitor_metric, factor=0.80, patience=25,
                                                      cooldown=10, verbose=verbose_val)
        self.red_lr_aggressive = callbacks.ReduceLROnPlateau(monitor=self.monitor_metric, factor=0.50, patience=50,
                                                             cooldown=10, verbose=verbose_val)
        self.csv_logger = callbacks.CSVLogger(self.loss_dir, append=True, separator=',')

        # Collection of the above callbacks.
        if bool(cbs):
            self.cbs = cbs
        elif self.disp_loss == True:
            self.cbs = [self.es, self.mc, self.b_r, self.tbg, self.red_lr_small, self.red_lr_big,
                        self.red_lr_aggressive,
                        self.csv_logger, dp.PlotLearning()]
        else:
            self.cbs = [self.es, self.mc, self.b_r, self.tbg, self.red_lr_small, self.red_lr_big,
                        self.red_lr_aggressive, self.csv_logger]

        # Create directories if they don't exist.
        dpp.directory_create(self.path_to_save)
        dpp.directory_create(self.path_to_save + 'Backup/')
        dpp.directory_create(self.logs_dir)
        dpp.directory_create(self.backup_dir)
        dpp.directory_create(self.path_to_save + 'history/')
        dpp.directory_create(self.path_to_save + 'images/')

    # DNN ML Models:
    def seq_dnn_ml(self):
        """
        Sequential ML Model using DNN.
        """

        # Input layer
        input_data = keras.Input(shape=(None, 1))
        model = None

        for i in range(self.num_ml_layers):
            if i == 0:
                model = layers.Dense(units=self.dnn_units, activation=self.activation_func)(input_data)
                model = layers.Dropout(rate=self.dropout_rate)(model)
            else:
                model = layers.Dense(units=self.dnn_units, activation=self.activation_func)(model)
                model = layers.Dropout(rate=self.dropout_rate)(model)

        # Final Conv1D to output correct number of samples.
        model = layers.Dense(units=1, activation=self.activation_func)(model)

        # Compiling the ml_model.
        ml_model = keras.Model(input_data, model)
        ml_model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.ml_model = ml_model
        self.ml_model_type = 'SEQ_DNN'

    # CNN ML Models:
    def seq_cnn_ml(self):
        """
        Sequential ML Model using CNN.
        """

        # Input layer
        input_data = keras.Input(shape=(None, 1))
        model = None

        for i in range(self.num_ml_layers):
            if i == 0:
                model = layers.Conv1D(filters=self.num_filters, kernel_size=self.kernel_size,
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value)(input_data)
            else:
                model = layers.Conv1D(filters=self.num_filters, kernel_size=self.kernel_size,
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value)(model)

        # Final Conv1D to output correct number of samples.
        model = layers.Conv1D(filters=1, kernel_size=self.kernel_size, activation='linear', padding='same')(model)

        # Compiling the ml_model.
        ml_model = keras.Model(input_data, model)
        ml_model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.ml_model = ml_model
        self.ml_model_type = 'SEQ_CNN'

    # CNN Variant ML Models (models with varying kernel and filter sizes for each layer):
    def seq_cnn_variant_ml(self):
        """
        Sequential variant ML Model using CNN.
        """

        # Input layer
        input_data = keras.Input(shape=(None, 1))
        model = None

        for i in range(self.num_ml_layers):
            # Sanity check number of filters:
            curr_num_filters = int(self.num_filters / 2 ** (i))
            if curr_num_filters < 2:
                curr_num_filters = 2

            if i == 0:
                model = layers.Conv1D(filters=int(curr_num_filters),
                                      kernel_size=int(self.kernel_size * 2 ** (i)),
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value,
                                      dilation_rate=self.max_dilation,
                                      name='CNNI' + str(i))(input_data)
                model = layers.Dense(units=(self.dnn_units / 2 ** (i)), activation=self.activation_func)(model)
            else:
                model = layers.Conv1D(filters=int(curr_num_filters),
                                      kernel_size=int(self.kernel_size * 2 ** (i)),
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value,
                                      dilation_rate=self.max_dilation,
                                      name='CNNI' + str(i))(model)

                model = layers.Dense(units=(self.dnn_units / 2 ** (i)),
                                     activation=self.activation_func,
                                     bias_regularizer=self.regularizer_func,
                                     kernel_regularizer=self.regularizer_func,
                                     activity_regularizer=self.regularizer_func,
                                     name='DENSEI' + str(i))(model)

        # Final Conv1D to output correct number of samples.
        model = layers.Conv1D(filters=1,
                              kernel_size=self.kernel_size,
                              activation='linear',
                              padding='same',
                              dilation_rate=self.max_dilation,
                              name='CNNI_FIN')(model)
        model = layers.Dense(units=1, activation='linear', name='DENSE_LIN')(model)

        # Compiling the ml_model.
        ml_model = keras.Model(input_data, model)
        ml_model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.ml_model = ml_model
        self.ml_model_type = 'SEQ_CNN_VAR'

    # CNN-LSTM ML Models:
    def seq_cnn_lstm_ml(self):
        """
        Sequential ML Model using CNN-LSTM.
        """

        # Input layer
        input_data = keras.Input(shape=(None, 1))
        model = None

        for i in range(self.num_ml_layers * 2):
            # Sanity check number of filters:
            curr_num_filters = int(self.num_filters / 2 ** (i))
            if curr_num_filters < 2:
                curr_num_filters = 2

            if i == 0:
                model = layers.Conv1D(filters=int(curr_num_filters),
                                      kernel_size=int(self.kernel_size * 2 ** (i)),
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value,
                                      bias_regularizer=self.regularizer_func,
                                      kernel_regularizer=self.regularizer_func,
                                      activity_regularizer=self.regularizer_func,
                                      dilation_rate=self.max_dilation,
                                      name='CNNI' + str(i))(input_data)
            else:
                model = layers.Conv1D(filters=int(curr_num_filters),
                                      kernel_size=int(self.kernel_size * 2 ** (i)),
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value,
                                      bias_regularizer=self.regularizer_func,
                                      kernel_regularizer=self.regularizer_func,
                                      activity_regularizer=self.regularizer_func,
                                      dilation_rate=self.max_dilation,
                                      name='CNNI' + str(i))(model)

            if (i + 1) % 2 == 0 and i != 0:
                model = layers.LSTM(units=self.lstm_units * (i + 1), return_sequences=True,
                                    activation='tanh',
                                    bias_regularizer=self.regularizer_func,
                                    kernel_regularizer=self.regularizer_func,
                                    activity_regularizer=self.regularizer_func,
                                    name='LSTMI' + str(i))(model)

                model = layers.Dense(units=self.dnn_units * (i + 1),
                                     activation=self.activation_func,
                                     bias_regularizer=self.regularizer_func,
                                     kernel_regularizer=self.regularizer_func,
                                     activity_regularizer=self.regularizer_func,
                                     name='DENSEI' + str(i))(model)

        model = layers.Conv1D(filters=1, kernel_size=self.kernel_size, activation='linear', padding='same',
                              name='CNN_FIN')(model)
        model = layers.Dense(units=1, activation='linear',
                             name='DENSE_LIN')(model)
        # Compiling the ml_model.
        ml_model = keras.Model(input_data, model)
        ml_model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.ml_model = ml_model
        self.ml_model_type = 'SEQ_CNN_LSTM'

    # TCN ML Models (primary model type used for inversion applications):
    # noinspection PyCallingNonCallable
    def seq_tcn_ml(self):
        """
        Sequential ML Model using TCN.
        """

        # Input layer
        input_data = keras.Input(shape=(None, 1))
        model = None

        # Calculate the dilations list for use in TCN
        dilations_values = [i for i in np.arange(2, self.max_dilation + 1) if (np.log(i) / np.log(2)).is_integer()]
        dilations_values = np.insert(dilations_values, 0, 1)
        dilations_list = list(dilations_values)
        dilations_list = [int(i) for i in dilations_list]

        for i in range(self.num_ml_layers):
            # Sanity check number of filters:
            curr_num_filters = int(self.num_filters / 2 ** (i))
            if curr_num_filters < 2:
                curr_num_filters = 2

            if i == 0:
                if self.use_encoder == False:
                    model = TCN(input_shape=(None, None, 1), nb_filters=int(curr_num_filters),
                                kernel_size=int(self.kernel_size * 2 ** (i)), return_sequences=True,
                                dilations=dilations_list, activation=self.activation_func,
                                padding='same', use_skip_connections=True, name='TCNI' + str(i))(input_data)
                else:
                    model = TCN(input_shape=(None, None, 1), nb_filters=int(curr_num_filters),
                                kernel_size=int(self.kernel_size * 2 ** (i)), return_sequences=True,
                                dilations=dilations_list, activation=self.activation_func,
                                padding='same', use_skip_connections=True, name='TCNI' + str(i))(model)

                model = layers.TimeDistributed(layers.Dense(units=self.dnn_units * (i + 1),
                                                            activation=self.activation_func,
                                                            bias_regularizer=self.regularizer_func,
                                                            kernel_regularizer=self.regularizer_func,
                                                            activity_regularizer=self.regularizer_func))(model)
            else:
                model = TCN(input_shape=(None, None, 1), nb_filters=int(curr_num_filters),
                            kernel_size=int(self.kernel_size * 2 ** (i)), return_sequences=True,
                            dilations=dilations_list, activation=self.activation_func,
                            padding='same', use_skip_connections=True, name='TCNI' + str(i))(model)
                model = layers.TimeDistributed(layers.Dense(units=self.dnn_units * (i + 1),
                                                            activation=self.activation_func,
                                                            bias_regularizer=self.regularizer_func,
                                                            kernel_regularizer=self.regularizer_func,
                                                            activity_regularizer=self.regularizer_func))(model)

        # Compiling the ml_model.
        model = layers.Dense(units=1, activation='linear')(model)
        ml_model = keras.Model(input_data, model)
        ml_model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.ml_model = ml_model
        self.ml_model_type = 'SEQ_TCN'

    # Autoencoder/Denoiser Models:

    # CNN based autoencoders:
    def seq_cnn_ae(self):
        """
        AE Model using CNN.
        """

        # Input layer
        input_data = keras.Input(shape=(None, 1))
        model = None

        for i in range(self.num_ml_layers):
            if i == 0:
                model = layers.Conv1D(filters=self.num_filters, kernel_size=(self.kernel_size + (i ** 2)),
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value)(input_data)
            else:
                model = layers.Conv1D(filters=self.num_filters, kernel_size=(self.kernel_size + (i ** 2)),
                                      activation=self.activation_func, padding='same',
                                      strides=self.pooling_value)(model)

        for i in range(self.num_ml_layers):
            model = layers.Conv1DTranspose(filters=self.num_filters, kernel_size=(self.kernel_size + (i ** 2)),
                                           activation=self.activation_func, padding='same',
                                           strides=self.pooling_value)(model)

        # Final Conv1D to output correct number of samples.

        model = layers.Conv1D(filters=self.dnn_units, kernel_size=(self.kernel_size + ((2 * self.num_ml_layers) ** 2)),
                              activation=self.activation_func, padding='same')(model)
        model = layers.Dense(units=1, activation='linear')(model)

        # Compiling the ml_model.
        ml_model = keras.Model(input_data, model)
        ml_model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.ml_model = ml_model
        self.ml_model_type = 'SEQ_CNN_AE'

    # TCN based autoencoders.
    # noinspection PyCallingNonCallable
    def seq_tcn_ae(self):
        """
        AE Model using TCN.
        """
        if self.transform_type == 'CWT':
            self.seq_cnn_ae()
            print('CWT is not supported with TCN ML architecture due to memory limits, running CNN ML instead.')
            return
        elif self.transform_type == 'SWT':
            print('Might run into issues while running TCN ML using SWT data, consider using raw data in this mode.')

        # Input layer
        input_data = keras.Input(shape=(None, 1))
        model = None

        # Calculate the dilations list for use in TCN
        dilations_values = [i for i in np.arange(2, self.max_dilation + 1) if (np.log(i) / np.log(2)).is_integer()]
        dilations_values = np.insert(dilations_values, 0, 1)
        dilations_list = list(dilations_values)
        dilations_list = [int(i) for i in dilations_list]

        for i in range(self.num_ml_layers):
            curr_num_filters = int(self.num_filters / (2 ** i))

            if curr_num_filters < 1:
                curr_num_filters = 1

            if i == 0:
                model = TCN(input_shape=(None, None, 1), nb_filters=curr_num_filters,
                            kernel_size=int(self.kernel_size * 2 ** (i)), return_sequences=True,
                            dilations=dilations_list, activation=self.activation_func,
                            padding='same', use_skip_connections=True, name='TCNI' + str(i))(input_data)
                model = layers.TimeDistributed(layers.Dense(units=self.dnn_units * (i + 1),
                                                            activation=self.activation_func,
                                                            bias_regularizer=self.regularizer_func,
                                                            kernel_regularizer=self.regularizer_func,
                                                            activity_regularizer=self.regularizer_func))(model)
                model = layers.MaxPooling1D(self.pooling_value, padding='same')(model)
            else:
                model = TCN(input_shape=(None, None, 1), nb_filters=curr_num_filters,
                            kernel_size=int(self.kernel_size * 2 ** (i)), return_sequences=True,
                            dilations=dilations_list, activation=self.activation_func,
                            padding='same', use_skip_connections=True, name='TCNI' + str(i))(model)
                model = layers.TimeDistributed(layers.Dense(units=self.dnn_units * (i + 1),
                                                            activation=self.activation_func,
                                                            bias_regularizer=self.regularizer_func,
                                                            kernel_regularizer=self.regularizer_func,
                                                            activity_regularizer=self.regularizer_func))(model)
                model = layers.MaxPooling1D(self.pooling_value, padding='same')(model)

        for i in range(self.num_ml_layers):
            model = TCN(input_shape=(None, None, 1), nb_filters=curr_num_filters,
                        kernel_size=int(self.kernel_size * 2 ** (i)), return_sequences=True,
                        dilations=dilations_list, activation=self.activation_func,
                        padding='same', use_skip_connections=True, name='TCNO' + str(i))(model)
            model = layers.TimeDistributed(layers.Dense(units=self.dnn_units * (i + 1),
                                                        activation=self.activation_func,
                                                        bias_regularizer=self.regularizer_func,
                                                        kernel_regularizer=self.regularizer_func,
                                                        activity_regularizer=self.regularizer_func))(model)
            model = layers.UpSampling1D(self.pooling_value)(model)

        # Compiling the ml_model.
        model = layers.Dense(units=1, activation='linear')(model)
        ml_model = keras.Model(input_data, model)
        ml_model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.ml_model = ml_model
        self.ml_model_type = 'SEQ_TCN_AE'

    # Function that trains the created self.ml.model from one of the previous functions.
    def fit_model(self, lr_callback=None):
        """
        Trains the ml_model model, need to run tcn/cnn ml_model first.
        """
        if self.num_workers == 1:
            set_use_multiprocessing = False
        else:
            set_use_multiprocessing = True

        if self.use_advanced_lr == True:
            self.dwell = Dwell_CB(model=self.ml_model, lr_factor=0.50, loss_factor=1.50, dwell=True, verbose=True)
            # self.clr = CLR_CB(model=self.ml_model, lr_factor=5.0, criteria_factor=0.25, verbose=True)
            self.clr = CLR_CB(model=self.ml_model, lr_factor=5.0, criteria_factor=0.10, verbose=True)
            curr_callbacks = self.cbs
            curr_callbacks.append(self.dwell)
            curr_callbacks.append(self.clr)
        else:
            curr_callbacks = self.cbs

        self.ml_model.fit(self.trainingDataGen, epochs=self.num_epochs, validation_data=self.validationDataGen,
                          use_multiprocessing=set_use_multiprocessing, workers=self.num_workers,
                          callbacks=curr_callbacks, verbose=self.verbose_val)

    # Used to save the generated model.
    def save_model(self):
        """
        Saves the ml_model model and weights using h5 format, need to run tcn/cnn ml_model and fit model first.
        """
        self.file_name = self.path_to_save + self.model_name + self.transform_type + '_' + self.currentDate + '_' + self.ml_model_type + '_saved_model.h5'
        self.ml_model.save(self.file_name)

    # Used to load the generated model if needed.
    def load_model(self, file_name: str = '', load_custom_objects=None, update_current_model: bool = False,
                   compile_flag: bool = True):
        """
        Loads the ml_model model and weights using h5 format, need to run tcn/cnn ml_model and fit model,
        and have model saved.
        """

        if load_custom_objects is None and self.load_custom_objects is None:
            load_custom_objects = {}
        else:
            load_custom_objects = self.load_custom_objects
        if file_name == '':
            file_name = self.file_name

        loaded_model = keras.models.load_model(file_name, custom_objects=load_custom_objects, compile=compile_flag)
        # e.g. custom_objects= {"root_mean_squared_error": dpp.root_mean_squared_error, "TCN": TCN}

        if update_current_model == True:
            self.ml_model = loaded_model
            self.ml_model_type = 'Loaded_Model'

        return loaded_model

    # Returns model as output
    def get_model(self) -> keras.Model:
        """
        Returns models with weights.
        """
        return self.ml_model

    # Plots model and provides summary of model parameters
    def plot_model(self):
        """
        Plots model and summary.
        """
        print(self.ml_model.summary())
        model_plot_name = self.path_to_save + self.model_name + self.transform_type + '_' + self.ml_model_type + '_' + 'plot.png'
        plot_model(self.ml_model, to_file=model_plot_name, show_shapes=True, show_layer_names=True)

    # predicts result based on the last epoch of the trained model.
    def predict_result(self, x) -> NDArray:
        """
        predict result and return it.
        """
        y_predicted = self.ml_model.predict(x)
        return y_predicted


# Dwell_CB & CLR_CB are custom keras callbacks to try to make the learning rate more dynamic and improve the training.
class Dwell_CB(keras.callbacks.Callback):
    def __init__(self, model, lr_factor, loss_factor, dwell, verbose):
        super(Dwell_CB, self).__init__()
        self.model = model
        self.initial_lr = float(
            tf.keras.backend.get_value(model.optimizer.lr))  # get the initial-learning rate and save it
        self.lowest_vloss = np.inf  # set the lowest validation loss to infinity initially
        self.best_weights = self.model.get_weights()  # set best weights to model's initial weights
        self.verbose = verbose
        self.best_epoch = 0
        self.dwell = dwell
        self.lr_factor = lr_factor
        self.loss_factor = loss_factor

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        if self.dwell:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))  # get the current learning rate
            vloss = logs.get('val_loss')  # get the validation loss for this epoch
            if vloss > self.lowest_vloss * self.loss_factor:
                self.model.set_weights(self.best_weights)
                new_lr = lr * self.lr_factor
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                if self.verbose:
                    print('\n model weights reset to best weights from epoch ', self.best_epoch + 1,
                          ' and reduced lr to ', new_lr, flush=True)
            elif vloss <= self.lowest_vloss:
                self.lowest_vloss = vloss
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch


class CLR_CB(keras.callbacks.Callback):
    def __init__(self, model, verbose, lr_factor=1.05, criteria_factor=0.10):
        super(CLR_CB, self).__init__()
        self.model = model
        self.initial_lr = float(
            tf.keras.backend.get_value(model.optimizer.lr))  # get the initial-learning rate and save it
        self.verbose = verbose
        self.lr_factor = lr_factor
        self.criteria_factor = criteria_factor

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))  # get the current learning rate

        if lr < self.criteria_factor * self.initial_lr:
            new_lr = lr * self.lr_factor
            self.initial_lr = new_lr
            if self.verbose:
                print('\n Current LR is ' + str(float(tf.keras.backend.get_value(self.model.optimizer.lr)))
                      + ' and is ' + str(100 * self.criteria_factor)
                      + '% of initial LR, thus CLR condition is met and new LR is ' + str(new_lr) +
                      ', and this new LR has been set as the new initial LR', flush=True)

            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)


# This function is responsible for creating deep learning models based on the BaseML Class above.
def create_model(path_to_save: str, trainingDataGen, validationDataGen, num_epochs: int, arch_type: str, num_lyr: int,
                 num_dnn: int, num_dil: int, num_filt: int, krnl_size: int, pool_val: int, num_lstm: int = 10,
                 learning_rate: float = 0.001, new_model: bool = True, network_type: str = 'SEQ',
                 model_update_path: str = '', custom_objects=None):
    """
    Function that handles the creation of deep learning models based on the BaseML Class.
    Arguments:
        path_to_save (str): the path to the directory to save the created model.
        trainingDataGen (customKerasGenerator): a custom Keras generator that is used to generate the training datasets.
        validationDataGen (customKerasGenerator):  a custom Keras generator that is used to generate the validation datasets.
        num_epochs (int): number of epochs that will be used for training
        arch_type (str): the architecture of the model to be created, can be DNN, CNN, CNN_VAR, CNN_LSTM, or TCN.
        num_lyr (int): number of layers for the network, for SEQ networks it is the same as the input, for AE networks, the actual value is double.
        num_dnn (int): number of neurons in the dense layers of the  network.
        num_dil (int): Value of max dilational value for TCNs.
        num_filt (int): Number of filters for convolutional layers.
        krnl_size (int): Kernel size for convolutional layers.
        pool_val (int): For AE networks, max pooling value for TCNs and stride for CNNs.
        num_lstm (int): Number of units in LSTM or CNN_LSTM networks.
        learning_rate (float): The initial learning rate for the network training.
        new_model (bool): Check as true if this is a model that is trained from scratch or False if it is a model that is loaded from a directory.
        network_type (str): The type of network being used either SEQ for sequential or AE for autoencoders.
        model_update_path (str): The path of model to be loaded if network_type is equal to False.
        custom_objects (customKerasObjects): CustomObjects to be loaded alongside the loaded model if network_type is equal to False.

    Returns:
        model: The created model.
        model_label: The model name.
    """

    # Name of the model
    model_label = arch_type + '_' + network_type + '_LYR' + str(num_lyr) + '_DNN' + str(num_dnn) + '_DIL' + \
                  str(num_dil) + '_FIL' + str(num_filt) + '_KRNL' + str(krnl_size) + '_POL' + str(pool_val) + '_LSTM' \
                  + str(num_lstm) + '_'

    # Prepare BaseMLClass with input parameters
    model = BaseMLClass(trainingDataGen, validationDataGen,
                        optimizer_func=optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
                        num_epochs=num_epochs, num_workers=1, loss_func=dpp.root_mean_squared_error,
                        num_ml_layers=num_lyr, dnn_units=num_dnn, max_dilation=num_dil,
                        num_filters=num_filt, kernel_size=krnl_size, pooling_value=pool_val,
                        lstm_units=num_lstm, disp_loss=False,
                        model_name=model_label, path_to_save=path_to_save,
                        load_custom_objects=custom_objects)

    print('*** Model ' + model.model_name + model.currentDate + ' has been created. ***')

    # Create the model based on network_type and arch_type
    if new_model == False:
        model.load_model(
            model_update_path, load_custom_objects=custom_objects, update_current_model=True)
    elif network_type == 'SEQ':
        if arch_type == 'TCN':
            model.seq_tcn_ml()
        elif arch_type == 'CNN':
            model.seq_cnn_ml()
        elif arch_type == 'CNN_VAR':
            model.seq_cnn_variant_ml()
        elif arch_type == 'CNN_LSTM':
            model.seq_cnn_lstm_ml()
        elif arch_type == 'DNN':
            model.seq_dnn_ml()
    elif network_type == 'AE':
        if arch_type == 'TCN':
            model.seq_tcn_ae()
        elif arch_type == 'CNN':
            model.seq_cnn_ae()

    model.plot_model()

    return model, model_label


# This function is responsible for training deep learning models based on the create_model function above.
def run_model(path_to_save: str, trainingDataGen, validationDataGen, x, y, num_epochs: int, arch_type: str,
              num_lyr: int, num_dnn: int, num_dil: int, num_filt: int, krnl_size: int, pool_val: int,
              num_lstm: int = 10,
              learning_rate: float = 0.0001, plot_fig: bool = True, new_model: bool = True, network_type: str = 'SEQ',
              model_update_path: str = '', custom_objects=None):
    """
    Function that handles the creation of deep learning models based on the BaseML Class.
    Arguments:
        path_to_save (str): the path to the directory to save the created model.
        trainingDataGen (customKerasGenerator): a custom Keras generator that is used to generate the training datasets.
        validationDataGen (customKerasGenerator):  a custom Keras generator that is used to generate the validation datasets.
        x: A single blind testing input dataset that can be plotted if plot_fig is set to True.
        y: A single blind testing output dataset that can be plotted if plot_fig is set to True.
        num_epochs (int): number of epochs that will be used for training
        arch_type (str): the architecture of the model to be created, can be DNN, CNN, CNN_VAR, CNN_LSTM, or TCN.
        num_lyr (int): number of layers for the network, for SEQ networks it is the same as the input, for AE networks, the actual value is double.
        num_dnn (int): number of neurons in the dense layers of the  network.
        num_dil (int): Value of max dilational value for TCNs.
        num_filt (int): Number of filters for convolutional layers.
        krnl_size (int): Kernel size for convolutional layers.
        pool_val (int): For AE networks, max pooling value for TCNs and stride for CNNs.
        num_lstm (int): Number of units in LSTM or CNN_LSTM networks.
        learning_rate (float): The initial learning rate for the network training.
        new_model (bool): Check as true if this is a model that is trained from scratch or False if it is a model that is loaded from a directory.
        plot_fig (bool): Plots a single blind testing dataset results (actual versus predicted).
        network_type (str): The type of network being used either SEQ for sequential or AE for autoencoders.
        model_update_path (str): The path of model to be loaded if network_type is equal to False.
        custom_objects (customKerasObjects): CustomObjects to be loaded alongside the loaded model if network_type is equal to False.
    Returns:
        min_val_loss: Minimum validation loss encountered
    """

    # Run create_model with inputs.
    model, model_label = create_model(path_to_save, trainingDataGen, validationDataGen, num_epochs, arch_type, num_lyr,
                                      num_dnn, num_dil, num_filt, krnl_size, pool_val, num_lstm, learning_rate,
                                      new_model, network_type, model_update_path, custom_objects)
    print('*** Model ' + model.model_name + model.currentDate + ' has started running. ***')

    # Start the training
    model.fit_model()

    # Obtains minimum validation loss and corresponding epoch
    loss_data = model.ml_model.history.history
    min_val_loss = min(loss_data.get('val_loss'))
    min_val_index = loss_data.get('val_loss').index(min_val_loss)
    min_train_loss = loss_data.get('loss')[min_val_index]
    min_epoch = min_val_index + 1

    # --- Predicting raw series using last model weights
    if plot_fig == True:
        y_hat = model.predict_result(x)
        dp.plot_ae_predict_raw(y, y_hat,
                               save_path=path_to_save + 'images/last_' + model.model_name + model.currentDate)

    # --- Predicting raw series using best model weights
    if plot_fig == True:
        y_hat = model.predict_best_result(x, load_custom_objects={
            "root_mean_squared_error": dpp.root_mean_squared_error, "TCN": TCN})
        dp.plot_ae_predict_raw(y, y_hat,
                               save_path=path_to_save + 'images/best_' + model.model_name + model.currentDate)

    # Save aggregate results for tested model.
    csv_data = [model_label, arch_type, num_lyr, num_dnn, num_dil, num_filt, krnl_size, pool_val, num_epochs,
                min_epoch, min_train_loss, min_val_loss]
    append_results_to_file(path_to_save, csv_data)

    print('*** Model ' + model.model_name + model.currentDate + ' has finished running. ***')

    return min_val_loss


# Initialize directory for saving summary of model results.
def init_results_file(path_to_save):
    """
        Function that handles the initialization of csv file for saving model summaries.
        Arguments:
            path_to_save (str): the path to the directory to save the summary csv file.
    """

    dpp.directory_create(path_to_save + 'Results/')

    header = ['Model Name', 'Type', 'LYR', 'DNN', 'DIL', 'FIL', 'KRNL', 'POL', 'Epochs', 'Min Epoch Loss',
              'Training Loss', 'Validation Loss']

    with open(path_to_save + 'Results/results.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)


def append_results_to_file(path_to_save, csv_data):
    """
        Function that appends summary results to the csv file with model summaries.
        Arguments:
            path_to_save (str): the path to the directory to with the summary csv file.
            csv_data (str): The data to be saved to the file.
    """
    dpp.directory_create(path_to_save + 'Results/')

    with open(path_to_save + 'Results/results.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow(csv_data)


# A simple sample main function to showcase how one will use the BaseMLClass to train a deep learning model.
def main():
    """
    Main function of dataMLs.
    """
    # AE CNN Testing
    num_datasets = 1000
    num_epochs = 500

    trainingDataGen = dpp.CustomDataGen(batch_size=1, n=num_datasets, gen_type='ML_VEL',
                                        transform_type='NA', scaling_flag=False, noisy_date=False)
    validationDataGen = dpp.CustomDataGen(batch_size=1, n=int(num_datasets * 0.3), gen_type='ML_VEL',
                                          transform_type='NA', scaling_flag=False, noisy_date=False)

    ml_model = BaseMLClass(trainingDataGen, validationDataGen,
                           optimizer_func=optimizers.Adam(learning_rate=0.0001, amsgrad=False),
                           num_epochs=num_epochs, num_workers=1, loss_func=dpp.root_mean_squared_error,
                           num_ml_layers=4, dnn_units=128, max_dilation=128, num_filters=64, kernel_size=8
                           , disp_loss=False)
    ml_model.seq_tcn_ml()
    ml_model.plot_model()
    ml_model.fit_model()

    # --- Retrieving one series
    dgc_predict = dg.DataGenerationClass(save_data_to_file=0)
    dgc_predict.generate_dataset()
    x, y = dgc_predict.retrieve_mult_input_output(x_data='CONV_REFL_CLEAN', y_data='IMP')

    # --- Predicting raw series:
    y_hat = ml_model.predict_result(x)
    dp.plot_ae_predict_raw(y, y_hat)


if __name__ == '__main__':
    main()
