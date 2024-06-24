import os
import numpy as np
import lmfit
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, ConstantModel, LinearModel
import matplotlib.pyplot as plt
# load folder with 2D integrations
path_fo_folder_2D_integs=r"d:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_stress_analysis_nca\output_folder_01\2D_integrations_numpy_arrays_intensity_txt"
# path to folder with q /radial coordinates
path_to_folder_radial_coord=r"d:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_stress_analysis_nca\output_folder_01\2D_integrations_numpy_arrays_radial_coord_txt"
# path to folder with azimuthal coordinates
path_to_folder_azimuth_coord=r"d:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_stress_analysis_nca\output_folder_01\2D_integrations_numpy_arrays_azimut_coord_txt"
# path to folder to save data
path_to_save_folder=r"d:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_stress_analysis_nca\save_folder_03"
# identify folder name for 2D integrations
last_folder_name_2D_integs=os.path.basename(path_fo_folder_2D_integs)
print(last_folder_name_2D_integs)
# identify folder name for radial coordinates
last_folder_name_radial_coord=os.path.basename(path_to_folder_radial_coord)
print(last_folder_name_radial_coord)
# identify folder name for azimuthal coordinates
last_folder_name_azimuth_coord=os.path.basename(path_to_folder_azimuth_coord)
print(last_folder_name_azimuth_coord)
# string for text extension
txt_extension="txt"
# string for npy extension
npy_extension="npy"
# root name of 2D integs, radial coordinates and azimuthal coordinates for evaluation
root_name="nca1_pristine"
# separator for 2D integs, radial coordinates and azimuthal coordinates for evaluation
separator="-"
# start number for evaluation
start_no=0
# stop / end number for evaluation
stop_no=91
# number of total evaluations
no_evals=stop_no-start_no+1
# number of digits in files numbering
no_digits=5
# set up fitting
model=LorentzianModel() + ConstantModel() # + LinearModel() # GaussianModel() # VoigtModel()
params=model.make_params(amplitude=30,center=2.06,sigma=0.01,c=20) # INITIAL PARAMETERS OF THE FIT GO HERE # center=  21.1 36.1 31.27 33.1 18.03 22.08 103.54
# minimum value of q range
q_min=2.0
# maximum value of q range
q_max=2.16
# number of fitting parameters
no_fitting_param=8
# define period for linear scanning in the sample
period=91
# index for start of linear scan
start_index=20
# index for stop / end of linear scan
stop_index=80
# main for evaluation loop
for index_0 in range(start_no,stop_no,period):
    # beginning index
    index_begin=start_index+index_0
    # end index
    index_end=stop_index+index_0
    if ((index_begin <= stop_no) & (index_end <= stop_no)):
        # second for loop for single linear scan
        for index_1 in range(stop_index-start_index+1):
            # current file number
            no_file=index_begin+index_1
            # convert file number to string
            no_file_str=str(no_file)
            # fill empty spaces in string with zeros
            no_file_str=no_file_str.rjust(no_digits,"0")
            # load all files for evaluation based on the extension text or numpy
            # load 2D integration
            # condition on importing 2D integration
            if (txt_extension in last_folder_name_2D_integs):
                # generate name of file
                name_2D_integ_file=root_name+separator+no_file_str+"."+txt_extension
                # generate path to 2D integration file
                path_to_2D_integ_file=os.path.join(path_fo_folder_2D_integs,name_2D_integ_file)
                # loda 2D integration data
                data_2D_integ=np.loadtxt(path_to_2D_integ_file)
            elif (npy_extension in last_folder_name_2D_integs):
                # generate name of file
                name_2D_integ_file=root_name+separator+no_file_str+"."+npy_extension
                # generate path to 2D integration file
                path_to_2D_integ_file=os.path.join(path_fo_folder_2D_integs,name_2D_integ_file)
                # load 2D integration data
                data_2D_integ=np.load(path_to_2D_integ_file)
            else:
                print("Unknown intensity data format.")
            # load radial coordinates
            # condition on importing radial coordinates
            if (txt_extension in last_folder_name_radial_coord):
                # generate name of file
                name_rad_coord_file=root_name+separator+no_file_str+"."+txt_extension
                # generate path to radial coordinate file
                path_to_rad_coord_file=os.path.join(path_to_folder_radial_coord,name_rad_coord_file)
                # load radial coordinate data
                data_rad_coord=np.loadtxt(path_to_rad_coord_file)
            elif (npy_extension in last_folder_name_radial_coord):
                # generate name of file
                name_rad_coord_file=root_name+separator+no_file_str+"."+npy_extension
                # generate path to radial coordinate file
                path_to_rad_coord_file=os.path.join(path_to_folder_radial_coord,name_rad_coord_file)
                # load radial coordinate data
                data_rad_coord=np.load(path_to_rad_coord_file)
            else:
                print("Unknown intensity data format.")
            # load azimuthal coordinates
            # condition on importing azimuthal coordinates
            if (txt_extension in last_folder_name_azimuth_coord):
                # generate name of file
                name_azimuth_coord_file=root_name+separator+no_file_str+"."+txt_extension
                # generate path to azimuthal coordinate file
                path_to_azimuth_coord_file=os.path.join(path_to_folder_azimuth_coord,name_azimuth_coord_file)
                # load azimuthal coordinate data
                data_azimuth_coord=np.loadtxt(path_to_azimuth_coord_file)
            elif (npy_extension in last_folder_name_azimuth_coord):
                # generate name of file
                name_azimuth_coord_file=root_name+separator+no_file_str+"."+npy_extension
                # generate path to azimuthal coordinate file
                path_to_azimuth_coord_file=os.path.join(path_to_folder_azimuth_coord,name_azimuth_coord_file)
                # load azmuthal coordinate data
                data_azimuth_coord=np.load(path_to_azimuth_coord_file)
            else:
                print("Unknown intensity data format.")
        ##    # print all current data
        ##    print(data_2D_integ)
        ##    print(data_rad_coord)
        ##    print(data_azimuth_coord)
            # transpose 2D integration
            data_2D_integ=np.transpose(data_2D_integ)
            #print(data_2D_integ)
            # get shapes of input arrays
            shape_2D_integ=data_2D_integ.shape
        ##    print(shape_2D_integ)
            shape_rad_coord=data_rad_coord.shape
        ##    print(shape_rad_coord)
            shape_azimuth_coord=data_azimuth_coord.shape
        ##    print(shape_azimuth_coord)
            # get number of radial points
            no_rad_points=shape_2D_integ[0]
            # get numebr of azimuthal coordinates
            no_azimuth_points=shape_2D_integ[1]
        ##    # print number of radial and azimuthal coordinates
        ##    print(no_rad_points)
        ##    print(no_azimuth_points)
            # initialize buffer to save fitting results
            results_buffer=np.zeros((no_azimuth_points,(no_fitting_param+2)),dtype=float)
            for index_2 in range(no_azimuth_points):
                # load radial or q data
                data_rad=data_rad_coord
                # load intensity data
                data_int=data_2D_integ[:,index_2]
                # load azimuthal ccordinate of current radial or q-cut
                data_azimuth=data_azimuth_coord[index_2]
                # select smaller part of data
                data_rad_part=data_rad[(data_rad>=q_min)&(data_rad<=q_max)]
                data_int_part=data_int[(data_rad>=q_min)&(data_rad<=q_max)]
                # perform fitting with suitable model
                results_fitting=model.fit(data_int_part,params,x=data_rad_part)
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
                results_buffer[index_2][0]=data_azimuth*(np.pi/180)
                results_buffer[index_2][1]=amplitude
                results_buffer[index_2][2]=amplitude_err
                results_buffer[index_2][3]=center
                results_buffer[index_2][4]=center_err
                results_buffer[index_2][5]=sigma
                results_buffer[index_2][6]=sigma_err
                results_buffer[index_2][7]=c
                results_buffer[index_2][8]=c_err
                results_buffer[index_2][9]=chi_square
##                # real-time plot of fitted peak function
##                plt.figure(1)
##                plt.clf()
##                plt.plot(data_rad_part,data_int_part,'o-')
##                plt.plot(data_rad_part,results_fitting.best_fit)
##                plt.pause(0.1)
            # save buffer with results
            # generate name of file with text and numpy extension
            name_save_result_txt="res_fit_"+root_name+separator+no_file_str+"."+txt_extension
            name_save_result_npy="res_fit_"+root_name+separator+no_file_str+"."+npy_extension
            # generate path to save file as text and numpy array
            path_to_save_file_txt=os.path.join(path_to_save_folder,name_save_result_txt)
            path_to_save_file_npy=os.path.join(path_to_save_folder,name_save_result_npy)
            # save results of fitting
            np.savetxt(path_to_save_file_txt,results_buffer,delimiter='\t',newline='\n')
            np.save(path_to_save_file_npy,results_buffer)
            
            
            
