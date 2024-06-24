# this program convert pyfai result matrix to matrix fitter 7
# import numpy
import numpy as np
# selecd data to work with as "numpy" or "text" data
data_switch = "numpy"
# specify path to result matrix
path_to_result_matrix = "d:/programs_work/Python/pyfai_integrate_test/test_data_from_poongodi/10-1-2024/output_folder_08/1D_integrations_numpy_arrays_intensity/live_00023.npy"
# specify path to x coordinates
path_to_result_matrix_x = "d:/programs_work/Python/pyfai_integrate_test/test_data_from_poongodi/10-1-2024/output_folder_08/1D_integrations_numpy_arrays_q_coord/live_00023.npy"
# specify path to output folder
path_to_output_folder = "d:/programs_work/Python/pyfai_integrate_test/test_data_from_poongodi/10-1-2024/output_folder_08/matrix_fitter_7/"
# specify root name for output
root_name_output = "live"
# name of result matrix in xy
name_result_matrix = root_name_output + ".txt"
# name of result matrix in x
name_result_matrix_x = root_name_output + "_x" + ".txt"
# path to output file for result matrix
path_to_output_file_result_matrix = path_to_output_folder + name_result_matrix
# path to output file for result matrix - x
path_to_output_file_result_matrix_x = path_to_output_folder + name_result_matrix_x
# main condition
if (data_switch == "numpy"):
    result_matrix = np.load(path_to_result_matrix)
    result_matrix = result_matrix.transpose()
    print(result_matrix)
    result_matrix_x = np.load(path_to_result_matrix_x)
    result_matrix_x = result_matrix_x[:,0]
    result_matrix_x = result_matrix_x.T
    print(result_matrix_x)
elif (data_switch == "text"):
    result_matrix = np.loadtxt(path_to_result_matrix)
    result_matrix = result_matrix.transpose()
    print(result_matrix)
    result_matrix_x = np.loadtxt(path_to_result_matrix_x)
    result_matrix_x = result_matrix_x[:,0]
    result_matrix_x = result_matrix_x.T
    print(result_matrix_x)
else:
    print("Do nothing")
# save result matrix and result matrix - x as text files
np.savetxt(path_to_output_file_result_matrix, result_matrix, delimiter='\t', newline='\n')
np.savetxt(path_to_output_file_result_matrix_x, result_matrix_x, delimiter='\t', newline = '\t')
# strip last tab character in result matrix - x 
# opened in text-mode; all EOLs are converted to '\n'
f = open(path_to_output_file_result_matrix_x, "r")
for line in f:
    line = line.rstrip('\t')
f.close()
f = open(path_to_output_file_result_matrix_x, "w")
f.write(line)
f.close()
    
    
     
