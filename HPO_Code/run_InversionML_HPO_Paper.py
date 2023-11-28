# A File utilizing other classes and functions to train a velocity inversion model from a seismic trace.
# This is a part of Automating Hyperparameter Optimization in Geophysics with Optuna: A Comparative Study paper
# by Hussain Almarzooq & Umair Bin Waheed

import io
import logging
import sys
import datetime
import uuid
from tcn import TCN
import dataGeneration as dg
import dataPreProcessing as dpp
import baseML as bml
import optuna


# Optuna Optimizer
def objective_inversion(trial_obj):
    # Trial Variables:

    path_to_save = 'HPO_Paper_Inversion_Model_Optimization_Optuna_TPE\\'

    arch_type = trial_obj.suggest_categorical("arch_type", ['DNN', 'CNN', 'CNN_VAR', 'CNN_LSTM', 'TCN'])
    num_lyr = trial_obj.suggest_int("num_layers", 1, 4)
    num_dnn = trial_obj.suggest_int("num_dnn", 2, 4096)
    num_filt = trial_obj.suggest_int("num_filt", 2, 128)
    krnl_size = trial_obj.suggest_int("krnl_size", 2, 64)
    num_dil = trial_obj.suggest_int("num_dil", 2, 128)
    num_lstm = trial_obj.suggest_int("num_lstm", 2, 64)
    learning_rate = trial_obj.suggest_float("Learning Rate", 0.000001, 0.001)
    pool_val = 1

    # Constant Variables:
    num_datasets = 250
    num_epochs = 30

    trainingDataGen = dpp.CustomDataGen(batch_size=1, n=num_datasets, gen_type='ML_VEL',
                                        transform_type='NA', scaling_flag=False, noisy_date=False)
    validationDataGen = dpp.CustomDataGen(batch_size=1, n=int(num_datasets * 0.3), gen_type='ML_VEL',
                                          transform_type='NA', scaling_flag=False, noisy_date=False)

    model, model_label = bml.create_model(path_to_save, trainingDataGen, validationDataGen, num_epochs, arch_type,
                                          num_lyr, num_dnn, num_dil, num_filt, krnl_size, pool_val, num_lstm=num_lstm,
                                          learning_rate=learning_rate, network_type='SEQ')

    print('*** Model ' + model.model_name + model.currentDate + ' has started running. ***')

    stream = io.StringIO()
    model.ml_model.summary(print_fn=lambda t: stream.write(t + '\n'))
    summary_string = stream.getvalue()
    stream.close()

    num_parameters_text = summary_string.partition("Total params: ")[2]
    num_parameters_text = num_parameters_text.partition('\n')[0]
    num_parameters = int(num_parameters_text.replace(',', ''))

    # Start the training
    if num_parameters <= 20000000 and (num_parameters <= 5000000 and arch_type != 'CNN_LSTM'):
        model.fit_model()

        # Obtains minimum validation loss and corresponding epoch
        loss_data = model.ml_model.history.history
        min_val_loss = min(loss_data.get('val_loss'))
        min_val_index = loss_data.get('val_loss').index(min_val_loss)
        min_train_loss = loss_data.get('loss')[min_val_index]
        min_epoch = min_val_index + 1

        csv_data = [model_label, arch_type, num_lyr, num_dnn, num_dil, num_filt, krnl_size, pool_val, num_epochs,
                    min_epoch, min_train_loss, min_val_loss]
        bml.append_results_to_file(path_to_save, csv_data)

        print('*** Model ' + model.model_name + model.currentDate + ' has finished running. ***')

        score = min_val_loss

    else:
        score = 100000.0

    return float(score)


def main():
    """
    Main function of dataMLs.
    """
    opt_method = 'final_model'  # optuna / optuna_resume / manual / final_model / model_update

    # Manual Optimization:
    if opt_method == 'manual':
        # AE CNN Testing
        num_datasets = 250
        num_epochs = 30

        trainingDataGen = dpp.CustomDataGen(batch_size=1, n=num_datasets, gen_type='ML_VEL',
                                            transform_type='NA', scaling_flag=False, noisy_date=False)
        validationDataGen = dpp.CustomDataGen(batch_size=1, n=int(num_datasets * 0.3), gen_type='ML_VEL',
                                              transform_type='NA', scaling_flag=False, noisy_date=False)

        # --- Retrieving one series
        dgc_predict = dg.DataGenerationClass(save_data_to_file=0)
        dgc_predict.generate_dataset()
        x, y = dgc_predict.retrieve_mult_input_output(x_data='CONV_REFL_CLEAN', y_data='VEL')

        path_to_save = 'HPO_Paper_Inversion_Model_Optimization_manual\\'
        bml.init_results_file(path_to_save)

        arch_type = ['DNN', 'CNN', 'CNN_VAR', 'CNN_LSTM', 'TCN']
        num_lyr = [1, 2, 3, 4]
        num_dnn = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096]
        num_dil = [2, 4, 8, 16, 32, 64, 128]
        num_filt = [2, 4, 8, 16, 32, 64, 128]
        krnl_size = [2, 4, 8, 16, 32, 64]
        num_lstm = [5, 10, 25, 50]
        learning_rate = [0.000001, 0.00001, 0.0001, 0.001]

        # Arch_type loop
        for i in range(len(arch_type)):
            curr_num_lyr = 1
            curr_dnn = 1024
            curr_num_filt = 64
            curr_krnl_size = 32
            curr_pool_val = 1
            curr_learning_rate = 0.0001

            if arch_type[i] == 'TCN' or arch_type[i] == 'CNN_VAR' or arch_type[i] == 'CNN_LSTM':
                curr_dill = 32
            else:
                curr_dill = 0

            if arch_type[i] == 'CNN_LSTM':
                curr_num_lstm = 10
            else:
                curr_num_lstm = 0

            bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, arch_type[i],
                          curr_num_lyr, curr_dnn, curr_dill, curr_num_filt, curr_krnl_size, curr_pool_val,
                          curr_num_lstm, network_type='SEQ', learning_rate=curr_learning_rate, plot_fig=False)

        # Number of layers loop
        for i in range(len(num_lyr)):
            # curr_num_lyr = 1
            curr_dnn = 1024
            curr_num_filt = 64
            curr_krnl_size = 32
            curr_pool_val = 1
            curr_learning_rate = 0.0001

            for j in range(len(arch_type)):
                if arch_type[j] == 'TCN' or arch_type[j] == 'CNN_VAR' or arch_type[j] == 'CNN_LSTM':
                    curr_dill = 32
                else:
                    curr_dill = 0

                if arch_type[j] == 'CNN_LSTM':
                    curr_num_lstm = 10
                else:
                    curr_num_lstm = 0

                if arch_type[j] == 'TCN':
                    bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, arch_type[j],
                                  num_lyr[i], curr_dnn, curr_dill, curr_num_filt, curr_krnl_size, curr_pool_val,
                                  curr_num_lstm, network_type='SEQ', learning_rate=curr_learning_rate, plot_fig=False)

        # Number of Dnns loop
        for i in range(len(num_dnn)):
            curr_num_lyr = 1
            # curr_dnn = 1024
            curr_num_filt = 64
            curr_krnl_size = 32
            curr_pool_val = 1
            curr_learning_rate = 0.0001

            for j in range(len(arch_type)):
                if arch_type[j] == 'TCN' or arch_type[j] == 'CNN_VAR' or arch_type[j] == 'CNN_LSTM':
                    curr_dill = 32
                else:
                    curr_dill = 0

                if arch_type[j] == 'CNN_LSTM':
                    curr_num_lstm = 10
                else:
                    curr_num_lstm = 0

                bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, arch_type[j],
                              curr_num_lyr, num_dnn[i], curr_dill, curr_num_filt, curr_krnl_size, curr_pool_val,
                              curr_num_lstm, network_type='SEQ', learning_rate=curr_learning_rate, plot_fig=False)

        # Number of filters loop
        for i in range(len(num_filt)):
            curr_num_lyr = 1
            curr_dnn = 1024
            # curr_num_filt = 64
            curr_krnl_size = 32
            curr_pool_val = 1
            curr_learning_rate = 0.0001

            for j in range(len(arch_type)):
                if arch_type[j] == 'TCN' or arch_type[j] == 'CNN_VAR' or arch_type[j] == 'CNN_LSTM':
                    curr_dill = 32
                else:
                    curr_dill = 0

                if arch_type[j] == 'CNN_LSTM':
                    curr_num_lstm = 10
                else:
                    curr_num_lstm = 0

                bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, arch_type[j],
                              curr_num_lyr, curr_dnn, curr_dill, num_filt[i], curr_krnl_size, curr_pool_val,
                              curr_num_lstm, network_type='SEQ', learning_rate=curr_learning_rate, plot_fig=False)

        # Number of kernels loop
        for i in range(len(krnl_size)):
            curr_num_lyr = 1
            curr_dnn = 1024
            curr_num_filt = 64
            # curr_krnl_size = 32
            curr_pool_val = 1
            curr_learning_rate = 0.0001

            for j in range(len(arch_type)):
                if arch_type[j] == 'TCN' or arch_type[j] == 'CNN_VAR' or arch_type[j] == 'CNN_LSTM':
                    curr_dill = 32
                else:
                    curr_dill = 0

                if arch_type[j] == 'CNN_LSTM':
                    curr_num_lstm = 10
                else:
                    curr_num_lstm = 0

                bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, arch_type[j],
                              curr_num_lyr, curr_dnn, curr_dill, curr_num_filt, krnl_size[i], curr_pool_val,
                              curr_num_lstm, network_type='SEQ', learning_rate=curr_learning_rate, plot_fig=False)

        # Number of dilations loop
        for i in range(len(num_dil)):
            curr_num_lyr = 1
            curr_dnn = 1024
            curr_num_filt = 64
            curr_krnl_size = 32
            curr_pool_val = 1
            curr_num_lstm = 0
            curr_arch_type = 'TCN'
            curr_learning_rate = 0.0001

            bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, curr_arch_type,
                          curr_num_lyr, curr_dnn, num_dil[i], curr_num_filt, curr_krnl_size, curr_pool_val,
                          curr_num_lstm, network_type='SEQ', learning_rate=curr_learning_rate, plot_fig=False)

        # Number of LSTMs loop
        for i in range(len(num_lstm)):
            curr_num_lyr = 1
            curr_dnn = 1024
            curr_num_filt = 64
            curr_krnl_size = 32
            curr_pool_val = 1
            curr_num_dil = 32
            curr_arch_type = 'CNN_LSTM'
            curr_learning_rate = 0.0001

            bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, curr_arch_type,
                          curr_num_lyr, curr_dnn, curr_num_dil, curr_num_filt, curr_krnl_size, curr_pool_val,
                          num_lstm[i], network_type='SEQ', learning_rate=curr_learning_rate, plot_fig=False)

        # learning rate loop
        for i in range(len(learning_rate)):
            curr_num_lyr = 1
            curr_dnn = 1024
            curr_num_filt = 64
            curr_krnl_size = 32
            curr_pool_val = 1
            # curr_learning_rate = 0.0001

            for j in range(len(arch_type)):
                if arch_type[j] == 'TCN' or arch_type[j] == 'CNN_VAR' or arch_type[j] == 'CNN_LSTM':
                    curr_dill = 32
                else:
                    curr_dill = 0

                if arch_type[j] == 'CNN_LSTM':
                    curr_num_lstm = 10
                else:
                    curr_num_lstm = 0

                bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, arch_type[j],
                              curr_num_lyr, curr_dnn, curr_dill, curr_num_filt, curr_krnl_size, curr_pool_val,
                              curr_num_lstm, network_type='SEQ', learning_rate=learning_rate[i], plot_fig=False)



    # Optuna Optimization:
    elif opt_method == 'optuna':
        path_to_save = 'HPO_Paper_Inversion_Model_Optimization_Optuna_TPE\\'
        bml.init_results_file(path_to_save)

        n_trials = 300
        currentDate = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        study_name = 'INV_Study_' + currentDate + '_' + str(uuid.uuid4())
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        storage_name = "sqlite:///{}{}.db".format(path_to_save, study_name)

        with open(path_to_save + study_name + '.txt', 'w') as f:
            f.write(storage_name)

        study = optuna.create_study(study_name=study_name, direction="minimize",
                                    storage=storage_name, load_if_exists=True)
        study.optimize(objective_inversion, n_trials=n_trials, catch=(RuntimeWarning,))

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    elif opt_method == 'optuna_resume':
        study_name = 'X'
        storage_name = 'sqlite:///HPO_Paper_Inversion_Model_Optimization_Optuna_TPE\X.db'
        n_trials = 500

        study = optuna.create_study(study_name=study_name, direction="minimize",
                                    storage=storage_name, load_if_exists=True)
        study.optimize(objective_inversion, n_trials=n_trials, catch=(RuntimeWarning,))

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # Manual Optimization:
    elif opt_method == 'final_model':
        # Final Model
        num_datasets = 250
        num_epochs = 100
        scaling_bool = False

        trainingDataGen = dpp.CustomDataGen(batch_size=1, n=num_datasets, gen_type='ML_VEL',
                                            transform_type='NA', scaling_flag=scaling_bool, noisy_date=False)
        validationDataGen = dpp.CustomDataGen(batch_size=1, n=int(num_datasets * 0.3), gen_type='ML_VEL',
                                              transform_type='NA', scaling_flag=scaling_bool, noisy_date=False)

        # --- Retrieving one series

        path_to_save = 'HPO_Paper_Inversion_Model_Final\\'
        bml.init_results_file(path_to_save)

        for i in range(0, 25):
            for run_num in range(1, 3):
                # Grid Search
                if run_num == 1:
                    arch_type = 'TCN'
                    num_lyr = 1
                    num_dnn = 3072
                    num_dil = 16
                    num_filt = 128
                    krnl_size = 64
                    pool_val = 1
                    learning_rate = 0.001

                # Optuna
                elif run_num == 2:
                    arch_type = 'TCN'
                    num_lyr = 1
                    num_dnn = 3824
                    num_dil = 108
                    num_filt = 46
                    krnl_size = 14
                    pool_val = 1
                    learning_rate = 0.0008

                model, model_label = bml.create_model(path_to_save, trainingDataGen, validationDataGen, num_epochs,
                                                      arch_type, num_lyr, num_dnn, num_dil, num_filt, krnl_size,
                                                      pool_val,
                                                      num_lstm=0, learning_rate=learning_rate, network_type='SEQ')

                print('*** Model ' + model.model_name + model.currentDate + ' has started running. ***')

                model.fit_model()

                # Obtains minimum validation loss and corresponding epoch
                loss_data = model.ml_model.history.history
                min_val_loss = min(loss_data.get('val_loss'))
                min_val_index = loss_data.get('val_loss').index(min_val_loss)
                min_train_loss = loss_data.get('loss')[min_val_index]
                min_epoch = min_val_index + 1

                csv_data = [model_label + str(run_num) + '_' + str(i), arch_type, num_lyr, num_dnn, num_dil, num_filt,
                            krnl_size, pool_val,
                            num_epochs, min_epoch, min_train_loss, min_val_loss]
                bml.append_results_to_file(path_to_save, csv_data)

                print('*** Model ' + model.model_name + model.currentDate + ' has finished running. ***')



    elif opt_method == 'model_update':
        # Model Update
        num_datasets = 1000
        num_epochs = 150
        scaling_bool = False

        trainingDataGen = dpp.CustomDataGen(batch_size=1, n=num_datasets, gen_type='ML_VEL',
                                            transform_type='NA', scaling_flag=scaling_bool, noisy_date=False)
        validationDataGen = dpp.CustomDataGen(batch_size=1, n=int(num_datasets * 0.3), gen_type='ML_VEL',
                                              transform_type='NA', scaling_flag=scaling_bool, noisy_date=False)

        # --- Retrieving one series
        dgc_predict = dg.DataGenerationClass(save_data_to_file=0)
        dgc_predict.generate_dataset()
        x, y = dgc_predict.retrieve_mult_input_output(x_data='CONV_REFL_CLEAN', y_data='VEL')

        if scaling_bool == True:
            x, x_scaler = dpp.normalize_data(x, -1, 1, -1, 1)  # test 0 to 1 range for input?
            y, y_scaler = dpp.normalize_data(y, 0, 30000, 0, 1)

        path_to_save = 'X\\'
        bml.init_results_file(path_to_save)

        model_path = 'X\TCN_SEQ_LYR3_DNN1024_DIL128_FIL64_KRNL24_POL1_LSTM10_NA_20230520_2359_best_model.h5'
        custom_objects = {"root_mean_squared_error": dpp.root_mean_squared_error, "TCN": TCN}

        arch_type = 'TCN'
        num_lyr = 3
        num_dnn = 1024
        num_dil = 128
        num_filt = 64
        krnl_size = 24
        pool_val = 1

        bml.run_model(path_to_save, trainingDataGen, validationDataGen, x, y, num_epochs, arch_type, num_lyr,
                      num_dnn, num_dil, num_filt, krnl_size, pool_val, new_model=False, model_update_path=model_path,
                      custom_objects=custom_objects, network_type='SEQ')


if __name__ == '__main__':
    main()
