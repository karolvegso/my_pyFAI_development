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
path_to_save_data=r"D:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_sterss_analysis_soft\save_folder_battery_8\linear_scan_battery_8_04"
# root name of linear scan
root_name_lin_scan="linear_scan"
# linear scan separator
separator_lin_scan="_"
# number of digits in number of linear scan
no_digits_lin_scan=5
# linear scan extension
ext_lin_scan_txt=".txt"
ext_lin_scan_npy=".npy"
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
# define period for linear scanning in the sample
period=161
# index for start of linear scan
start_index=70
# index for stop / end of linear scan
stop_index=80
# initialize counter
counter=0
# number of linear scans
no_lin_scans=int((no_int_meas_points-0)/period)
# number of measurements per linear scan
no_meas_lin_scan=stop_index-start_index+1
# total number of fits
total_no_fits=no_lin_scans*no_meas_lin_scan
# buffers for linear scans
# initialize amplitude buffer
results_buffer_all=np.zeros((total_no_fits,no_fitting_param),dtype=float)
# main for loop
for index_0 in range(0,no_int_meas_points,period):
    # beginning index
    index_begin=start_index+index_0
    # end index
    index_end=stop_index+index_0
    # initialize buffer to save fitting results
    results_buffer=np.zeros((no_fitting_param,(stop_index-start_index+1)),dtype=float)
    # generate name of linear scan
    lin_scan_no_str=str(counter)
    lin_scan_no_str=lin_scan_no_str.rjust(no_digits_lin_scan,"0")
    # generate full name of linear scan
    full_name_lin_scan_txt=root_name_lin_scan+separator_lin_scan+lin_scan_no_str+ext_lin_scan_txt
    full_name_lin_scan_npy=root_name_lin_scan+separator_lin_scan+lin_scan_no_str+ext_lin_scan_npy
    # generate full path to linear scan
    path_to_save_data_txt=os.path.join(path_to_save_data,full_name_lin_scan_txt)
    path_to_save_data_npy=os.path.join(path_to_save_data,full_name_lin_scan_npy)
    # check indexing condition
    if ((index_begin <= no_int_meas_points) & (index_end <= no_int_meas_points)):
        # second for loop for single linear scan
        for index_1 in range(stop_index-start_index+1):
            # define linear scan index
            linear_scan_index=index_begin+index_1
            #print(linear_scan_index)
            #print(linear_scan_index)
            # select one column in data for q and intensity
            q_values=data_q_buffer[:,linear_scan_index]
            int_values=data_int_buffer[:,linear_scan_index]
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
            results_buffer[0][index_1]=amplitude
            results_buffer[1][index_1]=amplitude_err
            results_buffer[2][index_1]=center
            results_buffer[3][index_1]=center_err
            results_buffer[4][index_1]=sigma
            results_buffer[5][index_1]=sigma_err
            results_buffer[6][index_1]=c
            results_buffer[7][index_1]=c_err
            results_buffer[8][index_1]=chi_square
            # fill fitting resulst to buffer for all data
            results_buffer_all[counter*no_meas_lin_scan+index_1][0]=amplitude
            results_buffer_all[counter*no_meas_lin_scan+index_1][1]=amplitude_err
            results_buffer_all[counter*no_meas_lin_scan+index_1][2]=center
            results_buffer_all[counter*no_meas_lin_scan+index_1][3]=center_err
            results_buffer_all[counter*no_meas_lin_scan+index_1][4]=sigma
            results_buffer_all[counter*no_meas_lin_scan+index_1][5]=sigma_err
            results_buffer_all[counter*no_meas_lin_scan+index_1][6]=c
            results_buffer_all[counter*no_meas_lin_scan+index_1][7]=c_err
            results_buffer_all[counter*no_meas_lin_scan+index_1][8]=chi_square
            # real-time plot of fitted peak function
            plt.figure(1)
            plt.clf()
            plt.plot(q_values_part,int_values_part,'o-')
            plt.plot(q_values_part,results_fitting.best_fit)
            plt.pause(0.1)
        # save results
        np.savetxt(path_to_save_data_txt,results_buffer,delimiter='\t',newline='\n')
        np.save(path_to_save_data_npy,results_buffer)
        # increase counter
        counter=counter+1
# save results for all data
path_to_save_data_all_txt=os.path.join(path_to_save_data,"fitting_results_all.txt")
path_to_save_data_all_npy=os.path.join(path_to_save_data,"fitting_results_all.npy") 
np.savetxt(path_to_save_data_all_txt,results_buffer_all,delimiter='\t',newline='\n')
np.save(path_to_save_data_all_npy,results_buffer_all)
# save each point in linear scan separatelly
# root name of single point file
root_name_single_point="single_point"
# separator for single point file
separator_single_point="_"
# number of digits in single point file numbering
no_digits_single_point=5
# extension for single point output file in text format
ext_single_point_txt=".txt"
# extension for single point output file in numpy format
ext_single_point_npy=".npy"
# main for loop to save each point in linear scan separatelly
for index_2 in range(no_meas_lin_scan):
    # initialize results buffer for single point in linear scan
    results_buffer_single_point=np.zeros((no_lin_scans,no_fitting_param),dtype=float)
    for index_3 in range(no_lin_scans):
        results_buffer_single_point[index_3,:]=results_buffer_all[index_3*no_meas_lin_scan+index_2,:]
    # save single point in linear scan
    # number of single point in linear scan
    single_point_no=start_index+index_2
    # convert number of single linear scan to string
    single_point_no_str=str(single_point_no)
    # fill string with zeros
    single_point_no_str=single_point_no_str.rjust(no_digits_single_point,"0")
    # full path to save single point in linear scan
    full_name_single_point_txt=root_name_single_point+separator_single_point+single_point_no_str+ext_single_point_txt
    full_name_single_point_npy=root_name_single_point+separator_single_point+single_point_no_str+ext_single_point_npy
    # save single point in linear scan
    path_to_save_data_single_point_txt=os.path.join(path_to_save_data,full_name_single_point_txt)
    path_to_save_data_single_point_npy=os.path.join(path_to_save_data,full_name_single_point_npy) 
    np.savetxt(path_to_save_data_single_point_txt,results_buffer_single_point,delimiter='\t',newline='\n')
    np.save(path_to_save_data_single_point_npy,results_buffer_single_point)
    
