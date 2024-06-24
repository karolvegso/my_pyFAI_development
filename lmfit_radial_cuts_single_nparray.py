import numpy as np
import lmfit
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, ConstantModel, LinearModel
import matplotlib.pyplot as plt
import os
# define path to intensity numpy array
path_to_int_data=r"D:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_sterss_analysis_soft\output_folder_battery8\1D_integrations_numpy_arrays_intensity_txt\battery8.txt"
# define path to q-values numpy array
path_to_q_data=r"D:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_sterss_analysis_soft\output_folder_battery8\1D_integrations_numpy_arrays_q_coord_txt\battery8.txt"
# path to folder to save data
path_to_save_data=r"D:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_sterss_analysis_soft\save_folder_battery_8"
# identify file name for intensity numpy array
name_int=os.path.basename(path_to_int_data)
# identify file name for q numpy array
name_q=os.path.basename(path_to_q_data)
# string for text extension
txt_extension="txt"
# string for npy extension
npy_extension="npy"
# condition on importing intensity data
if (txt_extension in name_int):
    data_int_buffer=np.loadtxt(path_to_int_data)
elif (npy_extension in name_int):
    data_int_buffer=np.load(path_to_int_data)
else:
    print("Unknown intensity data format.")
# condition on importing q data
if (txt_extension in name_q):
    data_q_buffer=np.loadtxt(path_to_q_data)
elif (npy_extension in name_q):
    data_q_buffer=np.load(path_to_q_data)
else:
    print("Unknown intensity data format.")
# get shape of intensity data
shape_int_data=data_int_buffer.shape
print(shape_int_data)
# get shape of q data
shape_q_data=data_q_buffer.shape
print(shape_q_data)
# get number of q points
no_q_points=shape_int_data[0]
# get number of intensity measurements
no_int_meas_points=shape_int_data[1]
# set up fitting
model=LorentzianModel() + ConstantModel() # + LinearModel() # GaussianModel() # VoigtModel()
params=model.make_params(amplitude=30,center=20.5,sigma=0.1,c=20) # INITIAL PARAMETERS OF THE FIT GO HERE # center=  21.1 36.1 31.27 33.1 18.03 22.08 103.54
q_min=20.8
q_max=21.6
# number of fitting parameters
no_fitting_param=9
# initialize buffer to save fitting results
results_buffer=np.zeros((no_fitting_param,no_int_meas_points),dtype=float)
# main for loop
for index_0 in range(no_int_meas_points):
    # select one column in data for q and intensity
    q_values=data_q_buffer[:,index_0]
    int_values=data_int_buffer[:,index_0]
    # select smaller part of data
    q_values_part=q_values[(q_values>=q_min)&(q_values<=q_max)]
    int_values_part=int_values[(q_values>=q_min)&(q_values<=q_max)]
    # perform fitting with suitable model
    results_fitting=model.fit(int_values_part,params,x=q_values_part)
    # print results of fitting
    #print(lmfit.fit_report(results_fitting))
    # get results of fitting
    amplitude=results_fitting.params['amplitude'].value
    amplitude_err=results_fitting.params['amplitude'].stderr
    center=results_fitting.params['center'].value
    center_err=results_fitting.params['center'].stderr
    sigma=results_fitting.params['sigma'].value
    sigma_err=results_fitting.params['sigma'].stderr
    c=results_fitting.params['c'].value
    c_err=results_fitting.params['c'].stderr
    chi_square=results_fitting.chisqr
    # set initial params of next fit as the resulting params of current fit
    params = model.make_params(amplitude=amplitude,center=center,sigma=sigma,c=c) # slope=slope, intercept=intercept
    # insert fitting results to buffer
    results_buffer[0][index_0]=amplitude
    results_buffer[1][index_0]=amplitude_err
    results_buffer[2][index_0]=center
    results_buffer[3][index_0]=center_err
    results_buffer[4][index_0]=sigma
    results_buffer[5][index_0]=sigma_err
    results_buffer[6][index_0]=c
    results_buffer[7][index_0]=c_err
    results_buffer[8][index_0]=chi_square
    # real-time plot of fitted peak function
    plt.figure(1)
    plt.clf()
    plt.plot(q_values_part,int_values_part,'o-')
    plt.plot(q_values_part,results_fitting.best_fit)
    plt.pause(0.1)
# save results
path_to_save_data_txt=os.path.join(path_to_save_data,"fitting_results.txt")
path_to_save_data_npy=os.path.join(path_to_save_data,"fitting_results.npy") 
np.savetxt(path_to_save_data_txt,results_buffer,delimiter='\t',newline='\n')
np.save(path_to_save_data_npy,results_buffer)
    
