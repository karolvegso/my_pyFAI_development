import numpy as np
import pandas as pd
import lmfit
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, ConstantModel, LinearModel
import matplotlib.pyplot as plt
import os
import time
# specify path to radial cuts
path_to_rad_cuts=r"d:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_sterss_analysis_soft\output_folder_battery8_radial\radial_cuts"
# path to folder to save data
path_to_save_data=r"D:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_sterss_analysis_soft\save_folder_battery_8"
# specify root name of radial cut
root_name="battery8"
# separator of radial cut
separator="-"
# extension of radial cut
extension=".dat"
# starting radial cut number
start_no=0
# final / last radial cut number
stop_no=100
# step in evaluation of radial cuts
step_eval=1
# calcualte number of evaluations
no_evals=int((stop_no - start_no) / step_eval) + 1
# number of digits
no_digits=5
# number of lines to skip in radial cut header file, starting from 0
skip_header_lines=22
# set up fitting
model=LorentzianModel() + ConstantModel() # + LinearModel() # GaussianModel() # VoigtModel()
params=model.make_params(amplitude=30,center=20.5,sigma=0.1,c=20) # INITIAL PARAMETERS OF THE FIT GO HERE # center=  21.1 36.1 31.27 33.1 18.03 22.08 103.54
q_min=20.8
q_max=21.6
# number of fitting parameters
no_fitting_param=9
# initialize buffer to save fitting results
results_buffer=np.zeros((no_fitting_param,no_evals),dtype=float)
# main for loop
for index_0 in range(no_evals):
    # create stroing buffers for q, I, sigma
    q_col=np.array([],dtype=float)
    int_col=np.array([],dtype=float)
    int_err_col=np.array([],dtype=float)
    # calculate radial cut number
    radial_cut_no=start_no+index_0*step_eval
    radial_cut_no_str=str(radial_cut_no)
    radial_cut_no_str=radial_cut_no_str.rjust(no_digits,"0")
    # generate radial cut full name
    radial_cut_name=root_name+separator+radial_cut_no_str+extension
    # generate path to current radial cut
    path_to_rad_cut=os.path.join(path_to_rad_cuts,radial_cut_name)
    #print(path_to_rad_cut)
    # read radial cut data, skip header
    rad_cut_data=pd.read_csv(path_to_rad_cut,sep='\t',skiprows=22,engine='python')
    # read values in radial cut
    current_data=rad_cut_data.values
    # number of radial cut points
    no_rad_cut_points=current_data.shape[0]
    #print(no_rad_cut_points)
    # split radial cut data to q, I, sigma values to three columns
    for index_1 in range(no_rad_cut_points):
        single_data_str=str(current_data[index_1])
        single_data_str_len=len(single_data_str)
        #print(single_data_str)
        #print(single_data_str_len)
        index_2=0
        number_count=0
        while (number_count < 3):
            current_symbol=single_data_str[index_2]
            next_symbol=single_data_str[index_2+1]
            if ((current_symbol == ' ') & ((next_symbol.isdigit()) or (next_symbol == '+') or (next_symbol == '-'))):
                number_count=number_count+1
                number_index=index_2+1
                number_symbol=single_data_str[index_2+1]
                number_str=""
                new_index_position=index_2+1
                while (number_symbol.isdigit() or number_symbol == '.' or number_symbol == 'e' or number_symbol == '+' or number_symbol == '-'):
                    number_str=number_str+number_symbol
                    new_index_position=new_index_position+1
                    number_symbol=single_data_str[new_index_position]
                index_2=new_index_position+1
                if (number_count == 1):
                    q_value=float(number_str)
                elif (number_count == 2):
                    intensity_value=float(number_str)
                else:
                    intensity_err=float(number_str)
            else:
                index_2=index_2+1
        # print values
        #print(q_value)
        #print(intensity_value)
        #print(intensity_err)
        q_col=np.append(q_col,q_value)
        int_col=np.append(int_col,intensity_value)
        int_err_col=np.append(int_err_col,intensity_err)
##    plt.plot(q_col,int_col)
##    plt.show()
    # select smaller part of data
    q_col_part=q_col[(q_col>=q_min)&(q_col<=q_max)]
    int_col_part=int_col[(q_col>=q_min)&(q_col<=q_max)]
    # perform fitting with suitable model
    results_fitting=model.fit(int_col_part,params,x=q_col_part)
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
    plt.plot(q_col_part,int_col_part,'o-')
    plt.plot(q_col_part,results_fitting.best_fit)
    plt.pause(0.1)
# save results
path_to_save_data_txt=os.path.join(path_to_save_data,"fitting_results.txt")
path_to_save_data_npy=os.path.join(path_to_save_data,"fitting_results.npy") 
np.savetxt(path_to_save_data_txt,results_buffer,delimiter='\t',newline='\n')
np.save(path_to_save_data_npy,results_buffer)
