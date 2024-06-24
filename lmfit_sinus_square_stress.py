import numpy as np
import os
import lmfit
from lmfit import Model, Parameters, create_params
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel,  ConstantModel,  LinearModel
import matplotlib.pyplot as plt
# sepcify path to fitting data
path_to_data_folder=r"d:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_stress_analysis_nca\save_folder_01_taken"
# path to folder to save data
path_to_save_folder=r"d:\programs_work\Python\SimonMicky\Karol_learning_process\Simons_stress_analysis_nca\save_folder_04_sin2"
# specify root name of fitting data file
root_name="res_fit_nca1_pristine"
# start file number
start_file_no=20
# stop file number
stop_file_no=80
# separator in file name
separator="-"
# number of digits in file numbering
no_digits=5
# file name extension
extension=".txt"
# period in evaluation, period for linear scan in battery
period=161
# total number of periods or number of linear scans in battery
no_periods=1
"The fitting function"
def sin2_plus_correction(psi, const, k, psi0_1, d, psi0_2):
    return const + k*np.sin(psi)**2 + d*np.cos(psi-psi0_2)
# set up fitting model
sin2_model = Model(sin2_plus_correction)
params = sin2_model.make_params(const=20.6, k=0.02,
								psi0_1=dict(value=0, min=-np.pi, max=np.pi, vary=False),
								d=dict(value=0, vary=True),
								psi0_2=dict(value=0, min=-np.pi, max=np.pi, vary=True)) # INITIAL PARAMETERS OF THE FITTING GO HERE
# genearte results fitting buffers for fitting parameters
results_buffer_const=np.array([],dtype=float)
results_buffer_k=np.array([],dtype=float)
results_buffer_d=np.array([],dtype=float)
results_buffer_psi0_1=np.array([],dtype=float)
results_buffer_psi0_2=np.array([],dtype=float)
results_buffer_chisqr=np.array([],dtype=float)
# main for loop going through periods
for index_0 in range(no_periods):
    # calcualte start index of evaluation
    start_index=start_file_no+index_0*period
    # calculate stop / last index of evaluation
    stop_index=stop_file_no+index_0*period
##    print(start_index)
##    print(stop_index)
    # second for loop going through linear scan
    for index_1 in range(start_index,stop_index+1,1):
        # genarate file name number
        file_name_number_str=str(index_1)
        # fill file name number with zeros
        file_name_number_str=file_name_number_str.rjust(no_digits,"0")
        # generate current file name
        file_name=root_name+separator+file_name_number_str+extension
        # genearte full path to fitting data
        full_path_to_file=os.path.join(path_to_data_folder,file_name)
        # load data according to file name extension
        # main condition
        if (extension == ".txt"):
            current_data=np.loadtxt(full_path_to_file)
        elif (extension == ".npy"):
            current_data=np.load(full_path_to_file)
        else:
            print("Not text format or numpy format.")
        # get chi values
        chi_values=current_data[:,0]
        # get center positions
        center_values=current_data[:,3]
        # get center errors
        center_err_values=current_data[:,4]
        # perform fit using sin2_plus_correction function
        res=sin2_model.fit(center_values,params,psi=chi_values)
        # get fitting results
        const=res.params['const'].value
        k=res.params['k'].value
        psi0_1=res.params['psi0_1'].value
        d=res.params['d'].value
        psi0_2=res.params['psi0_2'].value
        chisqr=res.chisqr
        # plot result
        plt.clf()
        plt.plot(chi_values,center_values, 'o')
        plt.plot(chi_values,res.best_fit)
        plt.ylim([2.05, 2.07])
        plt.title(f'Current file: {file_name}')
        plt.pause(0.1)
        # save results
        results_buffer_const=np.append(results_buffer_const,const)
        results_buffer_k=np.append(results_buffer_k,k)
        results_buffer_d=np.append(results_buffer_d,d)
        results_buffer_psi0_1=np.append(results_buffer_psi0_1,psi0_1)
        results_buffer_psi0_2=np.append(results_buffer_psi0_2,psi0_2)
        results_buffer_chisqr=np.append(results_buffer_chisqr,chisqr)
# write results as text files
full_path_to_save_file_const_txt=os.path.join(path_to_save_folder,"results_sinsqr_const.txt")
full_path_to_save_file_k_txt=os.path.join(path_to_save_folder,"results_sinsqr_k.txt")
full_path_to_save_file_d_txt=os.path.join(path_to_save_folder,"results_sinsqr_d.txt")
full_path_to_save_file_psi0_1_txt=os.path.join(path_to_save_folder,"results_sinsqr_psi0_1.txt")
full_path_to_save_file_psi0_2_txt=os.path.join(path_to_save_folder,"results_sinsqr_psi0_2.txt")
full_path_to_save_file_chisqr_txt=os.path.join(path_to_save_folder,"results_sinsqr_chisqr.txt")
np.savetxt(full_path_to_save_file_const_txt, results_buffer_const, delimiter='\t')
np.savetxt(full_path_to_save_file_k_txt, results_buffer_k, delimiter='\t')
np.savetxt(full_path_to_save_file_d_txt, results_buffer_d, delimiter='\t')
np.savetxt(full_path_to_save_file_psi0_1_txt, results_buffer_psi0_1, delimiter='\t')
np.savetxt(full_path_to_save_file_psi0_2_txt, results_buffer_psi0_2, delimiter='\t')
np.savetxt(full_path_to_save_file_chisqr_txt, results_buffer_chisqr, delimiter='\t')
# write results as numpy files
full_path_to_save_file_const_npy=os.path.join(path_to_save_folder,"results_sinsqr_const.npy")
full_path_to_save_file_k_npy=os.path.join(path_to_save_folder,"results_sinsqr_k.npy")
full_path_to_save_file_d_npy=os.path.join(path_to_save_folder,"results_sinsqr_d.npy")
full_path_to_save_file_psi0_1_npy=os.path.join(path_to_save_folder,"results_sinsqr_psi0_1.npy")
full_path_to_save_file_psi0_2_npy=os.path.join(path_to_save_folder,"results_sinsqr_psi0_2.npy")
full_path_to_save_file_chisqr_npy=os.path.join(path_to_save_folder,"results_sinsqr_chisqr.npy")
np.savetxt(full_path_to_save_file_const_npy, results_buffer_const, delimiter='\t')
np.savetxt(full_path_to_save_file_k_npy, results_buffer_k)
np.savetxt(full_path_to_save_file_d_npy, results_buffer_d)
np.savetxt(full_path_to_save_file_psi0_1_npy, results_buffer_psi0_1)
np.savetxt(full_path_to_save_file_psi0_2_npy, results_buffer_psi0_2)
np.savetxt(full_path_to_save_file_chisqr_npy, results_buffer_chisqr)
# reshape results buffer
results_buffer_const_reshaped=np.reshape(results_buffer_const,((stop_file_no-start_file_no+1), no_periods))
results_buffer_k_reshaped=np.reshape(results_buffer_k,((stop_file_no-start_file_no+1), no_periods))
results_buffer_d_reshaped=np.reshape(results_buffer_d,((stop_file_no-start_file_no+1), no_periods))
results_buffer_psi0_1_reshaped=np.reshape(results_buffer_psi0_1,((stop_file_no-start_file_no+1), no_periods))
results_buffer_psi0_2_reshaped=np.reshape(results_buffer_psi0_2,((stop_file_no-start_file_no+1), no_periods))
results_buffer_chisqr_reshaped=np.reshape(results_buffer_chisqr,((stop_file_no-start_file_no+1), no_periods))
# save reshaped buffer numpy arrays as text files
full_path_to_save_file_const_reshaped_txt=os.path.join(path_to_save_folder,"results_sinsqr_const_reshaped.txt")
full_path_to_save_file_k_reshaped_txt=os.path.join(path_to_save_folder,"results_sinsqr_k_reshaped.txt")
full_path_to_save_file_d_reshaped_txt=os.path.join(path_to_save_folder,"results_sinsqr_d_reshaped.txt")
full_path_to_save_file_psi0_1_reshaped_txt=os.path.join(path_to_save_folder,"results_sinsqr_psi0_1_reshaped.txt")
full_path_to_save_file_psi0_2_reshaped_txt=os.path.join(path_to_save_folder,"results_sinsqr_psi0_2_reshaped.txt")
full_path_to_save_file_chisqr_reshaped_txt=os.path.join(path_to_save_folder,"results_sinsqr_chisqr_reshaped.txt")
np.savetxt(full_path_to_save_file_const_reshaped_txt, results_buffer_const_reshaped, delimiter='\t')
np.savetxt(full_path_to_save_file_k_reshaped_txt, results_buffer_k_reshaped, delimiter='\t')
np.savetxt(full_path_to_save_file_d_reshaped_txt, results_buffer_d_reshaped, delimiter='\t')
np.savetxt(full_path_to_save_file_psi0_1_reshaped_txt, results_buffer_psi0_1_reshaped, delimiter='\t')
np.savetxt(full_path_to_save_file_psi0_2_reshaped_txt, results_buffer_psi0_2_reshaped, delimiter='\t')
np.savetxt(full_path_to_save_file_chisqr_reshaped_txt, results_buffer_chisqr_reshaped, delimiter='\t')
# save reshaped buffer numpy arrays as numpy files
full_path_to_save_file_const_reshaped_npy=os.path.join(path_to_save_folder,"results_sinsqr_const_reshaped.npy")
full_path_to_save_file_k_reshaped_npy=os.path.join(path_to_save_folder,"results_sinsqr_k_reshaped.npy")
full_path_to_save_file_d_reshaped_npy=os.path.join(path_to_save_folder,"results_sinsqr_d_reshaped.npy")
full_path_to_save_file_psi0_1_reshaped_npy=os.path.join(path_to_save_folder,"results_sinsqr_psi0_1_reshaped.npy")
full_path_to_save_file_psi0_2_reshaped_npy=os.path.join(path_to_save_folder,"results_sinsqr_psi0_2_reshaped.npy")
full_path_to_save_file_chisqr_reshaped_npy=os.path.join(path_to_save_folder,"results_sinsqr_chisqr_reshaped.npy")
np.savetxt(full_path_to_save_file_const_reshaped_npy, results_buffer_const_reshaped)
np.savetxt(full_path_to_save_file_k_reshaped_npy, results_buffer_k_reshaped)
np.savetxt(full_path_to_save_file_d_reshaped_npy, results_buffer_d_reshaped)
np.savetxt(full_path_to_save_file_psi0_1_reshaped_npy, results_buffer_psi0_1_reshaped)
np.savetxt(full_path_to_save_file_psi0_2_reshaped_npy, results_buffer_psi0_2_reshaped)
np.savetxt(full_path_to_save_file_chisqr_reshaped_npy, results_buffer_chisqr_reshaped)        
