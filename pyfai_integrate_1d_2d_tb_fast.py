# import module for azimuthal integration
# import module for X-ray images opening
import pyFAI, fabio
# import module for displaying image
import matplotlib
from matplotlib import pyplot as plt
# import string module
import string
# import time module
import time
# import jupyter
from pyFAI.gui import jupyter
# import os
import os
# import numpy
import numpy as np
# print pyFAI version
print("pyFAI version is: ", pyFAI.version)
# insert path to folder with images
path_to_images = "d:/GIWAXS_measurements/Xray_Braun_glovebox/29-05-2024/Temperature study/selected_images_25_150degC/"
# specify root name of single image
image_root_name = "live"
# specify image name separator
image_name_separator = "_"
# specify image name extension
image_name_extension = ".tif"
# sepcify first image number
image_no_start = 0
# specify last image number
image_no_stop = 12
# specify image number step (step in time binning)
image_no_step = 1
# calculate number of image calculations
no_image_calc = (image_no_stop - image_no_start + 1) / image_no_step
no_image_calc = int(no_image_calc)
# number of digits in image numbering
no_image_digits = 5
# insert path to pyFAI calibration poni file
path_to_calib = "D:/GIWAXS_measurements/Xray_Braun_glovebox/29-05-2024/calib_pyfai_01/lizrfecl_pyfai_calib.poni"
# load pyFAI calibration poni file
ai = pyFAI.load(path_to_calib)
# specify size of images
# number of pixels in x /  horizontal dierction or number of columns
dim_x = 487
# number of pixels in y / vertical direction or number of rows
dim_y = 407
# insert path to pyFAI mask file
path_to_mask = "D:/GIWAXS_measurements/Xray_Braun_glovebox/29-05-2024/calib_pyfai_01/lizrfecl_pyfai_calib.edf"
# tell to program if you want to use mask, it can be True or False boolean value
mask_switch = True
if (mask_switch == True):
    mask_img = fabio.open(path_to_mask)
else:
    print("Mask is not used")
# specify number of points in radial cut
no_points_in_radial_cut = 1024
# specify units in which you want to perform azimuthal cut ("2th_deg" or "r_mm" or "q_nm^-1")
unit_of_radial_cut = "q_nm^-1"
# specify path to folder to save azimuthal cuts
path_to_output_folder = "d:/GIWAXS_measurements/Xray_Braun_glovebox/29-05-2024/output_folder_02/"
# create output sub folders for radial cuts and 2D integartions
path_to_output_folder_radial_cuts = path_to_output_folder + "/radial_cuts/"
# paths to 1D integrations
path_to_output_folder_1D_integs_numpy_arrays_intensity = path_to_output_folder + "/1D_integrations_numpy_arrays_intensity/"
path_to_output_folder_1D_integs_numpy_arrays_intensity_txt = path_to_output_folder + "/1D_integrations_numpy_arrays_intensity_txt/"
path_to_output_folder_1D_integs_numpy_arrays_q_coord = path_to_output_folder + "/1D_integrations_numpy_arrays_q_coord/"
path_to_output_folder_1D_integs_numpy_arrays_q_coord_txt = path_to_output_folder + "1D_integrations_numpy_arrays_q_coord_txt/"
path_to_output_folder_1D_integs_numpy_arrays_tth_coord = path_to_output_folder + "/1D_integrations_numpy_arrays_twotheta_coord/"
path_to_output_folder_1D_integs_numpy_arrays_tth_coord_txt = path_to_output_folder + "1D_integrations_numpy_arrays_twotheta_coord_txt/"
path_to_output_folder_1D_integs_numpy_arrays_sigma = path_to_output_folder + "/1D_integrations_numpy_arrays_sigma/"
path_to_output_folder_1D_integs_numpy_arrays_sigma_txt = path_to_output_folder + "/1D_integrations_numpy_arrays_sigma_txt/"
# paths to 2D integrations
path_to_output_folder_2D_integs_images = path_to_output_folder + "/2D_integrations_images/"
path_to_output_folder_2D_integs_numpy_arrays_intensity = path_to_output_folder + "/2D_integrations_numpy_arrays_intensity/"
path_to_output_folder_2D_integs_numpy_arrays_intensity_txt = path_to_output_folder + "/2D_integrations_numpy_arrays_intensity_txt/"
path_to_output_folder_2D_integs_numpy_arrays_radial_coord = path_to_output_folder + "/2D_integrations_numpy_arrays_radial_coord/"
path_to_output_folder_2D_integs_numpy_arrays_radial_coord_txt = path_to_output_folder + "/2D_integrations_numpy_arrays_radial_coord_txt/"
path_to_output_folder_2D_integs_numpy_arrays_azimut_coord = path_to_output_folder + "/2D_integrations_numpy_arrays_azimut_coord/"
path_to_output_folder_2D_integs_numpy_arrays_azimut_coord_txt = path_to_output_folder + "/2D_integrations_numpy_arrays_azimut_coord_txt/"
# create new output folders
os.mkdir(path_to_output_folder_radial_cuts)
# create new output folders for 2D integrations
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_intensity)
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_intensity_txt)
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_q_coord)
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_q_coord_txt)
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_tth_coord)
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_tth_coord_txt)
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_sigma)
os.mkdir(path_to_output_folder_1D_integs_numpy_arrays_sigma_txt)
# create new output folders for 1D integrations
os.mkdir(path_to_output_folder_2D_integs_images)
os.mkdir(path_to_output_folder_2D_integs_numpy_arrays_intensity)
os.mkdir(path_to_output_folder_2D_integs_numpy_arrays_intensity_txt)
os.mkdir(path_to_output_folder_2D_integs_numpy_arrays_radial_coord)
os.mkdir(path_to_output_folder_2D_integs_numpy_arrays_radial_coord_txt)
os.mkdir(path_to_output_folder_2D_integs_numpy_arrays_azimut_coord)
os.mkdir(path_to_output_folder_2D_integs_numpy_arrays_azimut_coord_txt)
# aziumthal cut name extension specified as dat file
radial_cut_name_extension = ".dat"
### specify variance ndarray, array containing the variance of the data, use None if you don't know
##variance_selector = None
# specify error model for calculation variance e.g. None or "poisson" or "azimuthal" (use defaultly "poisson")
error_model_selector = "poisson"
# specify radial range, ((float, float), optional), if not used, use None or ( , )
# the lower and upper range of the radial unit. If not provided, range is simply (min, max). Values outside the range are ignored.
radial_range_selector = [3.4 , 31.2]
# specify azimuthal range, ((float, float), optional), if not used, use None or ( , )
# the lower and upper range of the azimuthal angle in degree. If not provided, range is simply (min, max). Values outside the range are ignored.
azimuth_range_selector  = [-105.0 , -56.0]
# specify polarization factor, if experiment was done in laboratory, use None
# polarization factor between -1 (vertical) and +1 (horizontal). 0 for circular polarization or random, None for no correction, True for using the former correction
polarization_factor_selector = 0.0
### specify float value of a normalization monitor, use None (no correction)
##normalization_factor_selector = None
### dark (ndarray) – dark noise image
##dark_selector = None
### flat (ndarray) – flat field image
flat_selector = None
# method (IntegrationMethod) – IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
method_selector = ("full", "histogram", "cython")
# input data for 2D integration
# number of radial points
npt_rad_2D = no_points_in_radial_cut
# number of azimuthal points or chi points
npt_azim_2D = 360
# extension of 2D integration image
integ_2D_name_extension = ".edf"
# select radial range for 2D integration
radial_range_selector_2D = [0.34 , 3.12]
# select azimuthal range for 2D integration
azimuth_range_selector_2D = [0, 360]
# select method for 2D integration
method_selector_2D = "bbox"
# select unit for radial 2D integration
# Output units, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for now
unit_selector_2D = "q_A^-1"
# create numpy array to save time-resolved q-cuts
intensity_1d_integs = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_integs_q_coord = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_integs_tth_coord = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_integs_sigma = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
# main for loop
for index_0 in range(no_image_calc):
    print(index_0)
    # starting image number for time binning
    image_number_tb_start = image_no_start + image_no_step * index_0
##    print(image_number_tb_start)
    # last / stop / final image for time binning
    image_number_tb_stop = image_number_tb_start + image_no_step
##    print(image_number_tb_stop)
    # initialize numpy array for time binning
    img_tb = np.zeros((dim_y, dim_x), dtype=float)
    # for loop for time binning
    for index_1 in range(image_number_tb_start, image_number_tb_stop, 1):
        # calculate current image number for time binning
        image_number_tb_current = index_1
##        print(image_number_tb_current)
        # cast time binning current image number to string
        image_number_tb_current_str = str(image_number_tb_current)
##        print(image_number_tb_current_str)
        # right paddding of string with zeros
        image_number_tb_current_str = image_number_tb_current_str.rjust(no_image_digits, '0')
##        print(image_number_tb_current_str)
        # generate full current image name
        image_name_tb = image_root_name + image_name_separator + image_number_tb_current_str + image_name_extension
        print(image_name_tb)
        # generate full path to current image
        path_to_image_tb_full = path_to_images + image_name_tb
##        print(path_to_image_tb_full)
        # open single image using fabio module
        img = fabio.open(path_to_image_tb_full)
        # print dimensions of single image
##        print("The dimensions of image is: ", img.data.shape)
        # print image header
##        print("Print single image header: ", img.header)
        # print raw image data
##        print("Print raw image data or numpy matrix representing image: ", img.data)
##        # display image
##        plt.imshow(img.data, interpolation='nearest')
##        plt.show(block=False)
##        # wait one second to display image
##        plt.pause(1)
##        # close image
##        plt.close()
        # perform time binning
        img_tb = img_tb + img.data
##    # display image
##    plt.imshow(img_tb, interpolation='nearest')
##    plt.show(block=False)
##    # wait one second to display image
##    plt.pause(1)
##    # close image
##    plt.close()
    # calculate current image number
    image_number_current = index_0
##    print(image_number_current)
    # cast current image number to string
    image_number_current_str = str(image_number_current)
##    print(image_number_current_str)
    # right paddding of string with zeros
    image_number_current_str = image_number_current_str.rjust(no_image_digits, '0')
##    print(image_number_current_str)
    # specify radial cut name
    radial_cut_name = image_root_name + image_name_separator + image_number_current_str
##    print(radial_cut_name)
    # specify path to radial cut
    path_to_radial_cut = path_to_output_folder_radial_cuts + radial_cut_name + radial_cut_name_extension
    path_to_1D_integs_numpy_arrays_intensity = path_to_output_folder_1D_integs_numpy_arrays_intensity + radial_cut_name + ".npy"
    path_to_1D_integs_numpy_arrays_q_coord = path_to_output_folder_1D_integs_numpy_arrays_q_coord + radial_cut_name + ".npy"
    path_to_1D_integs_numpy_arrays_tth_coord = path_to_output_folder_1D_integs_numpy_arrays_tth_coord + radial_cut_name
    path_to_1D_integs_numpy_arrays_sigma = path_to_output_folder_1D_integs_numpy_arrays_sigma + radial_cut_name + ".npy"
    path_to_1D_integs_numpy_arrays_intensity_txt = path_to_output_folder_1D_integs_numpy_arrays_intensity_txt + radial_cut_name + ".txt"
    path_to_1D_integs_numpy_arrays_q_coord_txt = path_to_output_folder_1D_integs_numpy_arrays_q_coord_txt + radial_cut_name + ".txt"
    path_to_1D_integs_numpy_arrays_tth_coord_txt = path_to_output_folder_1D_integs_numpy_arrays_tth_coord_txt + radial_cut_name
    path_to_1D_integs_numpy_arrays_sigma_txt = path_to_output_folder_1D_integs_numpy_arrays_sigma_txt + radial_cut_name + ".txt"
    # perform q-cut or 2theta-cut
    if (mask_switch == True):
        if (unit_of_radial_cut == "q_nm^-1"):
            q, intensity, sigma = ai.integrate1d_ng(img_tb, npt = no_points_in_radial_cut, unit=unit_of_radial_cut, error_model=error_model_selector,
                                                    radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                    mask = mask_img.data,
                                                    polarization_factor = polarization_factor_selector, 
                                                    method = method_selector, 
                                                    filename=path_to_radial_cut)
##            print(q)
##            print(intensity)
            intensity_1d_integs[:, index_0] = intensity
            intensity_1d_integs_q_coord[:, index_0] = q
            intensity_1d_integs_sigma[:, index_0] = sigma
            # save all data from 1D integrations as numpy arrays
            np.save(path_to_1D_integs_numpy_arrays_intensity, intensity_1d_integs)
            np.save(path_to_1D_integs_numpy_arrays_q_coord, intensity_1d_integs_q_coord)
            np.save(path_to_1D_integs_numpy_arrays_sigma, intensity_1d_integs_sigma)
            # save all data from 1D integartions as text files
            np.savetxt(path_to_1D_integs_numpy_arrays_intensity_txt, intensity_1d_integs, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_q_coord_txt, intensity_1d_integs_q_coord, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_sigma_txt, intensity_1d_integs_sigma, delimiter='\t', newline='\n')
        else:
            twotheta, intensity, sigma = ai.integrate1d_ng(img_tb, npt = no_points_in_radial_cut, unit=unit_of_radial_cut,  error_model=error_model_selector,
                                                           radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                           mask = mask_img.data,
                                                           polarization_factor = polarization_factor_selector, 
                                                           method = method_selector, 
                                                           filename=path_to_azimuhal_cut)
##            print(twotheta)
##            print(intensity)
            intensity_1d_integs[:, index_0] = intensity
            intensity_1d_integs_tth_coord[:, index_0] = twotheta
            intensity_1d_integs_sigma[:, index_0] = sigma
            # save all data from 1D integrations as numpy arrays
            np.save(path_to_1D_integs_numpy_arrays_intensity, intensity_1d_integs)
            np.save(path_to_1D_integs_numpy_arrays_tth_coord, intensity_1d_integs_tth_coord)
            np.save(path_to_1D_integs_numpy_arrays_sigma, intensity_1d_integs_sigma)
            # save all data from 1D integartions as text files
            np.savetxt(path_to_1D_integs_numpy_arrays_intensity_txt, intensity_1d_integs, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_tth_coord_txt, intensity_1d_integs_tth_coord, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_sigma_txt, intensity_1d_integs_sigma, delimiter='\t', newline='\n')
    else:
        if (unit_of_radial_cut == "q_nm^-1"):
            q, intensity, sigma = ai.integrate1d_ng(img_tb, npt = no_points_in_radial_cut, unit=unit_of_radial_cut, error_model=error_model_selector,
                                                    radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                    polarization_factor = polarization_factor_selector, 
                                                    method = method_selector, 
                                                    filename=path_to_radial_cut)
##            print(q)
##            print(intensity)
            intensity_1d_integs[:, index_0] = intensity
            intensity_1d_integs_q_coord[:, index_0] = q
            intensity_1d_integs_sigma[:, index_0] = sigma
            # save all data from 1D integrations as numpy arrays
            np.save(path_to_1D_integs_numpy_arrays_intensity, intensity_1d_integs)
            np.save(path_to_1D_integs_numpy_arrays_q_coord, intensity_1d_integs_q_coord)
            np.save(path_to_1D_integs_numpy_arrays_sigma, intensity_1d_integs_sigma)
            # save all data from 1D integartions as text files
            np.savetxt(path_to_1D_integs_numpy_arrays_intensity_txt, intensity_1d_integs, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_q_coord_txt, intensity_1d_integs_q_coord, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_sigma_txt, intensity_1d_integs_sigma, delimiter='\t', newline='\n')
        else:
            twotheta, intensity, sigma = ai.integrate1d_ng(img_tb, npt = no_points_in_radial_cut, unit=unit_of_radial_cut,  error_model=error_model_selector,
                                                           radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                           polarization_factor = polarization_factor_selector, 
                                                           method = method_selector, 
                                                           filename=path_to_azimuhal_cut)
##            print(twotheta)
##            print(intensity)
            intensity_1d_integs[:, index_0] = intensity
            intensity_1d_integs_tth_coord[:, index_0] = twotheta
            intensity_1d_integs_sigma[:, index_0] = sigma
            # save all data from 1D integrations as numpy arrays
            np.save(path_to_1D_integs_numpy_arrays_intensity, intensity_1d_integs)
            np.save(path_to_1D_integs_numpy_arrays_tth_coord, intensity_1d_integs_tth_coord)
            np.save(path_to_1D_integs_numpy_arrays_sigma, intensity_1d_integs_sigma)
            # save all data from 1D integartions as text files
            np.savetxt(path_to_1D_integs_numpy_arrays_intensity_txt, intensity_1d_integs, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_tth_coord_txt, intensity_1d_integs_tth_coord, delimiter='\t', newline='\n')
            np.savetxt(path_to_1D_integs_numpy_arrays_sigma_txt, intensity_1d_integs_sigma, delimiter='\t', newline='\n')
            
##    if (unit_of_azimcut == "q_nm^-1"):
##        q, intensity, sigma = ai.integrate1d_ng(img_tb, no_points_in_cut, correctSolidAngle=True, unit=unit_of_azimcut, variance=variance_selector, error_model=error_model_selector,
##                                                radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
##                                                polarization_factor = polarization_factor_selector, normalization_factor = normalization_factor_selector,
##                                                dark = dark_selector, flat = flat_selector, method = method_selector, 
##                                                filename=path_to_azimuhal_cut)
##    else:
##        twotheta, intensity, sigma = ai.integrate1d_ng(img_tb, no_points_in_cut, correctSolidAngle=True, unit=unit_of_azimcut, variance=variance_selector, error_model=error_model_selector,
##                                                       radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
##                                                       polarization_factor = polarization_factor_selector, normalization_factor = normalization_factor_selector,
##                                                       dark = dark_selector, flat = flat_selector, method = method_selector, 
##                                                       filename=path_to_azimuhal_cut)
    # perorm 2D integration in q vs chi plot
    # specify 2D integration name
    integ_2D_name = image_root_name + image_name_separator + image_number_current_str
    # specify path to 2D integration image
    path_to_integ_2D_images = path_to_output_folder_2D_integs_images + integ_2D_name + integ_2D_name_extension
    path_to_2D_integs_numpy_arrays_intensity = path_to_output_folder_2D_integs_numpy_arrays_intensity + integ_2D_name + ".npy"
    path_to_2D_integs_numpy_arrays_intensity_txt = path_to_output_folder_2D_integs_numpy_arrays_intensity_txt + integ_2D_name + ".txt"
    path_to_2D_integs_numpy_arrays_radial_coord = path_to_output_folder_2D_integs_numpy_arrays_radial_coord + integ_2D_name + ".npy"
    path_to_2D_integs_numpy_arrays_radial_coord_txt = path_to_output_folder_2D_integs_numpy_arrays_radial_coord_txt + integ_2D_name + ".txt"
    path_to_2D_integs_numpy_arrays_azimut_coord = path_to_output_folder_2D_integs_numpy_arrays_azimut_coord + integ_2D_name + ".npy"
    path_to_2D_integs_numpy_arrays_azimut_coord_txt = path_to_output_folder_2D_integs_numpy_arrays_azimut_coord_txt + integ_2D_name + ".txt"
    if (mask_switch == True):
        intensity_integ_2D, radial, azimuthal, sigma_integ_2D = ai.integrate2d_ng(img_tb, npt_rad = npt_rad_2D, npt_azim = npt_azim_2D, filename = path_to_integ_2D_images,
                       error_model=error_model_selector,
                       mask = mask_img.data,  
                       polarization_factor = polarization_factor_selector,
                       method = method_selector_2D, unit = unit_selector_2D)
##        # display image
##        plt.imshow(intensity_integ_2D, interpolation='nearest')
##        plt.show(block=False)
##        # wait one second to display image
##        plt.pause(1)
##        # close image
##        plt.close()
        # save all data from 2D integrations as numpy arrays
        np.save(path_to_2D_integs_numpy_arrays_intensity, intensity_integ_2D)
        np.save(path_to_2D_integs_numpy_arrays_radial_coord, radial)
        np.save(path_to_2D_integs_numpy_arrays_azimut_coord, azimuthal)
        # save all data from 2D integartions as text files
        np.savetxt(path_to_2D_integs_numpy_arrays_intensity_txt, intensity_integ_2D, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_numpy_arrays_radial_coord_txt, radial, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_numpy_arrays_azimut_coord_txt, azimuthal, delimiter='\t', newline='\n')
    else:
        intensity_integ_2D, radial, azimuthal = ai.integrate2d_ng(img_tb, npt_rad = npt_rad_2D, npt_azim = npt_azim_2D, filename = path_to_integ_2D_images,
                       error_model=error_model_selector,
                       polarization_factor = polarization_factor_selector,
                       method = method_selector_2D, unit = unit_selector_2D)
##        # display image
##        plt.imshow(intensity_integ_2D, interpolation='nearest')
##        plt.show(block=False)
##        # wait one second to display image
##        plt.pause(1)
##        # close image
##        plt.close()
        # save all data from 2D integrations as numpy arrays
        np.save(path_to_2D_integs_numpy_arrays_intensity, intensity_integ_2D)
        np.save(path_to_2D_integs_numpy_arrays_radial_coord, radial)
        np.save(path_to_2D_integs_numpy_arrays_azimut_coord, azimuthal)
        # save all data from 2D integartions as text files
        np.savetxt(path_to_2D_integs_numpy_arrays_intensity_txt, intensity_integ_2D, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_numpy_arrays_radial_coord_txt, radial, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_numpy_arrays_azimut_coord_txt, azimuthal, delimiter='\t', newline='\n')
        
##    ai.integrate2d_ng(img_tb, npt_rad = npt_rad_2D, npt_azim = npt_azim_2D, filename = path_to_integ_2D,
##                   error_model=error_model_selector,
##                   radial_range = radial_range_selector_2D, azimuth_range = azimuth_range_selector_2D,
##                   polarization_factor = polarization_factor_selector,
##                   method = method_selector_2D, unit = unit_selector_2D)
        
