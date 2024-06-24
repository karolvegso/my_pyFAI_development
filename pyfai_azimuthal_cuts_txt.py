# this program performs chi-cuts from 2D integartion data got by pyFAI
# import numpy
import numpy as np
# path to 2D integration images in text format
path_to_2D_integs = "d:/programs_work/Python/pyfai_integrate_test/test_data_from_poongodi/10-1-2024/output_folder_08/2D_integrations_numpy_arrays_intensity_txt/"
# path to radial coordinates in text format
path_to_rad_coord = "d:/programs_work/Python/pyfai_integrate_test/test_data_from_poongodi/10-1-2024/output_folder_08/2D_integrations_numpy_arrays_radial_coord_txt/live_00023.txt"
# path to azimuthal coordinates in text format
path_to_azimuth_coord = "d:/programs_work/Python/pyfai_integrate_test/test_data_from_poongodi/10-1-2024/output_folder_08/2D_integrations_numpy_arrays_azimut_coord_txt/live_00023.txt"
# path to folder to save output
path_to_output_folder = "d:/programs_work/Python/pyfai_integrate_test/test_data_from_poongodi/10-1-2024/output_folder_10_azimuth/"
# root name of output file
output_file_root_name = "azimuthal_cuts"
# output file name as numpy array
output_file_name_npy = output_file_root_name + ".npy"
# output file name as text file
output_file_name_txt = output_file_root_name + ".txt"
# output file for azimuthal coordinates as numpy array
output_file_name_azimuth_coord_npy =output_file_root_name + "_azimuth_coord" + ".npy"
# output file for azimuthal coordinates as tetxt file
output_file_name_azimuth_coord_txt =output_file_root_name + "_azimuth_coord" + ".txt"
# path to output file as numpy
path_to_output_file_npy = path_to_output_folder + output_file_name_npy
# path to output file as text
path_to_output_file_txt = path_to_output_folder + output_file_name_txt
# path to output file for azimuthal coordinates as numpy array
path_to_output_file_azimuth_coord_npy = path_to_output_folder + output_file_name_azimuth_coord_npy
# path to output file for azimuthal coordinates as text file
path_to_output_file_azimuth_coord_txt = path_to_output_folder + output_file_name_azimuth_coord_txt
# root name of 2D integartion images
root_name = "live"
# separator in name of 2D integ images
separator = "_"
# specify number of image digits in numbering
number_image_digits = 5
# start number of 2D integ images
start_number = 0
# stop / last / final number of 2D images
stop_number = 23
# number of azimuthal cuts
number_of_azimuth_cuts = (stop_number - start_number) + 1
# load numpy array with radial coordinates
rad_coord = np.loadtxt(path_to_rad_coord)
print(rad_coord)
# number of radial points
no_rad_points = rad_coord.shape[0]
print(no_rad_points)
# load numpy array with azimuthal coordinates
azimuth_coord = np.loadtxt(path_to_azimuth_coord)
print(azimuth_coord)
# number of azimuthal points
no_azimuth_points = azimuth_coord.shape[0]
print(no_azimuth_points)
# specify minimum value of q radial coordinate
q_min = 1.50
# specify maximum value of q radial coordinate
q_max = 1.53
# find indices of experimental points in the q range
q_indices = np.where((rad_coord > q_min) & (rad_coord < q_max))
print(q_indices)
# take first index in the radial or q integration range
q_index_min = q_indices[0][0]
print(q_index_min)
# take last index in the radial or q integration range
q_index_max = q_indices[0][-1]
print(q_index_max)
# initialize output azimuthal cuts numpy array
output_azimuth_cuts = np.zeros((no_azimuth_points, number_of_azimuth_cuts), dtype = float)
# main for loop to perform azimuthal cuts
for index_0 in range(number_of_azimuth_cuts):
    # current 2D integration image number
    number_current = start_number + index_0
    # convert 2D integartion image number to string
    number_current_str = str(number_current)
    number_current_str = number_current_str.rjust(number_image_digits, '0')
    # generate name of 2D integration image
    name_2D_integ_image = root_name + separator + number_current_str + ".txt"
    # print name of 2D integartion image in numpy format
    print(name_2D_integ_image)
    # genearte path to 2D integration image
    path_to_2D_integ_image = path_to_2D_integs + name_2D_integ_image
    # print path to single 2D integartion image
    print(path_to_2D_integ_image)
    # load 2D integration image
    image_numpy = np.loadtxt(path_to_2D_integ_image)
    # print 2D integration
    print(image_numpy)
    # get shape of 2D integartion image
    image_nummpy_shape = image_numpy.shape
    print(image_nummpy_shape)
    # initilaze numpy array for current azimuthal cut
    azimuth_cut_current = np.zeros((no_azimuth_points, 1), dtype = float)
    # perform integration of azimutahl cuts in q or radial coordinate direction
    # main integartion for loop in q or ardial coordinate direction
    for index_1 in range(q_index_min, (q_index_max + 1), 1):
        azimuth_cut_current[:,0] = azimuth_cut_current[:,0] + image_numpy[:,index_1]
    # print current azimuthal cut
    print(azimuth_cut_current)
    # insert current azimutha cuts to output numpy array representing all azimuthal cuts
    output_azimuth_cuts[:,index_0] = azimuth_cut_current[:,0]
# save output numpy array filled with azimuthal cuts as numpy array
np.save(path_to_output_file_npy, output_azimuth_cuts)
# save output numpy array filled with azimuthal cuts as text file
np.savetxt(path_to_output_file_txt, output_azimuth_cuts, delimiter='\t', newline='\n')
# save azimutal coordinates as numpy array
np.save(path_to_output_file_azimuth_coord_npy, azimuth_coord)
# save azimutal coordinates as tetx file
np.savetxt(path_to_output_file_azimuth_coord_txt, azimuth_coord, delimiter='\t', newline='\n')
