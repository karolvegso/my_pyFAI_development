import numpy as np
import lmfit
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, ConstantModel, LinearModel, ExponentialModel
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
params=model.make_params(amplitude=10,center=21.2,sigma=0.1,c=0) # INITIAL PARAMETERS OF THE FIT GO HERE # center=  21.1 36.1 31.27 33.1 18.03 22.08 103.54
# take data from q minimum
q_min=20.7
# take data from q maximum
q_max=21.7
# background subtraction instructions
# take some background points from left
# number of background points from left
no_bg_points_left=3
# take some background points from right
# number of background points from right
no_bg_points_right=3
# set up fitting for background
# set up fitting
model_bg=LinearModel() # LinearModel() # ExponentialModel() + ConstantModel
# initial fitting parameters for exponential model with constant
#params_bg=model_bg.make_params(amplitude=1,decay=1,c=0)
# initial fitting parameters for linear model
params_bg=model_bg.make_params(slope=0,intercept=1)
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
    # select bg data - q values
    q_values_part_bg=q_values_part[0:no_bg_points_left]
    q_values_part_bg=np.append(q_values_part_bg,q_values_part[(-no_bg_points_right):-1])
    q_values_part_bg=np.append(q_values_part_bg,q_values_part[-1])
    # select bg data - intensity values
    int_values_part_bg=int_values_part[0:no_bg_points_left]
    int_values_part_bg=np.append(int_values_part_bg,int_values_part[(-no_bg_points_right):-1])
    int_values_part_bg=np.append(int_values_part_bg,int_values_part[-1])
    # perform fitting with background model
    results_fitting_bg=model_bg.fit(int_values_part_bg,params_bg,x=q_values_part_bg)
    # real-time plot of fitted peak function
    plt.figure(1)
    plt.clf()
    plt.plot(q_values_part,int_values_part,'o-')
    plt.plot(q_values_part_bg,results_fitting_bg.best_fit)
    plt.pause(1.0)
    # get background function fitting parameters
    slope=results_fitting_bg.params['slope'].value
    intercept=results_fitting_bg.params['intercept'].value
    # ampltude_exp=results_fitting_bg.params['amplitude'].value
    # decay_exp=results_fitting_bg.params['decay'].value
    # c_exp=results_fitting_bg.params['c'].value
    # linear background calculation
    lin_bg=slope*q_values_part[:]+intercept
    # lin_bg=amplitude_exp*np.exp((-1.0)*(q_values_part[:]/decay_exp))+c_exp
    # subtract linear backround
    int_values_part=int_values_part[:]-lin_bg[:]
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
    plt.pause(1.0)
# save results
path_to_save_data_txt=os.path.join(path_to_save_data,"fitting_results.txt")
path_to_save_data_npy=os.path.join(path_to_save_data,"fitting_results.npy") 
np.savetxt(path_to_save_data_txt,results_buffer,delimiter='\t',newline='\n')
np.save(path_to_save_data_npy,results_buffer)
    
