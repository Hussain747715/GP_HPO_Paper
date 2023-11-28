# GP_HPO_Paper
Data sample, results, and code associated with Geophysical Prospecting Hyperparameter Optimization Paper 


# 	Computational Environment

This section of the appendix will briefly cover the computational environment that was used to write and run the code. It will be split into hardware and software sections.


# •	Hardware Environment
All the codes, plots, and results were created on an Asus Zephyrus G14 GA401QM (2021) laptop with the below specifications. 

Item	Specification

Processor	AMD Ryzen™ 9 5900HS Mobile Processor

Graphics	NVIDIA® GeForce RTX™ 3060 Laptop GPU

GPU Memory	6GB GDDR6

RAM Memory	32GB DDR4

Storage Capacity	4TB


# •	Software Environment
The operating system used was Windows 11 (kept up-to-date) with the main integrated development environment (IDE) in use throughout being Pycharm 2021-2022. The entirety of the code was written for Python 3.9 and was accelerated using CUDA V11.7.64. The following libraries with their respective versions necessary for the environment setup are shown below.
Library Purpose

Computational	Graphical	Deep Learning	Other

Library	Version

bruges	0.5.4	

matplotlib	3.7.0	

keras	2.10.0	

ipython	8.10.0

numpy	1.24.2	

optuna-dashboard	0.9.0	

keras-tcn	3.5.0	

nptyping	2.4.1

pandas	1.5.3	

seaborn	0.12.2	

optuna	3.1.0	 

PyWavelets (pywt)	1.4.1	 	

scikit-learn	1.2.1	

scipy	1.10.0		

tensorflow	2.10.0	



# Code High-Level Walkthrough


# •	Data Generation
All the synthetic data generation and augmentation for this thesis is done inside the dataGeneration.py file. The most important portion of this file is the DataGenerationClass class which is responsible for creating random lithological sequences, adding layers based on appropriate lithologies for each sequence. These layers are then filled with appropriate velocity and density values based on empirical relationships for each lithology. Next, the acoustic impedance and subsequently the reflectivity series are calculated. Finally, a wavelet is convolved with the reflectivity series in addition to some noise to produce the synthetic corridor stack (seismic trace). 

# •	Data Pre-Processing
Similar to the way the data generation functionality was handled, one file is primarily responsible for the data pre-processing prior to input to deep learning models which is dataPreProcessing.py file. This file is composed of one class, CustomDataGen. The CustomDataGen class is responsible for using the DataGenerationClass to create datasets and feeding them directly to the deep learning models without saving them to disk first.

# •	Data Plotting & Results Plotting
The next two files dataPlotting.py and resultsPlotting.py have many of the underlying basic plotting functions designed for this thesis.

Function Name	Purpose

font_sizes_defaults	Loads default font sizes for plots (used in other plots).

max_axis_value	Determines appropriate max axis value.

plot_model	Plots lithology, velocity, density, and acoustic impledance in depth.

plot_xplot	Plots a crossplot of velocity versus density colored by lithology or acoustic impedance.

plot_refl	Plots a reflectivtiy series, convolved data, noisy data, and data with gap.

plot_refl_compare	Plots a comparison between data convolved with the three wavelet options.

plot_cwt_real_imag	Plots the real and imaginary components of a CWT dataset.

plot_cwt_magn	Plots the magnitude of a CWT dataset.

plot_cwt_magn_phase	Plots the magnitude and phase of a CWT dataset.

plot_mra	Plots a DWT/SWT dataset.

plot_ae_predict_raw	Plots the actual and reconstructed trace.

plot_ae_mra_predict	Plots the actual and reconstructed DWT/SWT dataset.

plot_cwt_magn_predict	Plots the actual and reconstructed CWT dataset.

PlotLearning	Plots the objective function and learning rate at epoch end.


# •	Base Deep Learning Model
The baseML.py is composed of 3 classes and 4 functions. The classes are BaseMLClass, Dwell_CB, and CLR_CB. The first one is the one responsible for constructing, training, and predicting deep learning models while the latter two are custom callbacks related to implementing a dynamic learning rate while training. The four available functions are create_model, run_model, init_results_file, and append_results_to_file. The first two functions use the BaseMLClass to create and train deep learning models while the latter two are used within BaseMLClass to create .csv files to store training history data.

 
# •	Creating, Running, and Updating Models
The final group of code files is responsible for using all aforementioned codes in order to create the specific deep learning models shown in this thesis. Below is a table with the file name and the models that were created from said file.

File	Input Type	Purpose

run_DenoisingML_HPO_Paper.py	Seismic Trace	Denoising seismic traces.

run_InversionML_HPO_Paper.py	Seismic Trace	Inverting for velocity.
