import os
import fabio
from fabio.TiffIO import TiffIO

# path to h5 file
path_to_h5_file = "d:/programs_work/Python/pyfai_integrate_test/python_pyfai_test_05/h5_file_example/sample_water0000.h5"

# slit path to h5 file
path_to_folder, file_name_h5 = os.path.split(path_to_h5_file)
print(path_to_folder)
print(file_name_h5)
root_file_name = file_name_h5.split(".h5")
root_file_name = root_file_name[0]
print(root_file_name)
separator = "_"
no_digits = 5

# create output folders to save coneversion results of h5 file 
path_to_output_folder_first_frame_of_h5_file = path_to_folder + "/" + "first_frame_of_h5_file_as_tiff"
path_to_output_folder_multiple_tif_files = path_to_folder + "/" + "multiple_tiff_files"
os.mkdir(path_to_output_folder_first_frame_of_h5_file)
os.mkdir(path_to_output_folder_multiple_tif_files)

img = fabio.open(path_to_h5_file)
print(f"This file contains {img.nframes} frames of shape {img.shape} and weights {os.stat(path_to_h5_file).st_size/1e6:.3f} MB")

#Initialization of the TiffIO file with the first frame, note the mode "w"
# write first frame or zero-th frame of h5 file
file_name_first_frame_tif = file_name_h5.replace(".h5", ".tiff")
path_to_save_first_frame_tif_file = path_to_output_folder_first_frame_of_h5_file + "/" + file_name_first_frame_tif
img_tif = TiffIO(path_to_save_first_frame_tif_file, mode='w')
img_tif.writeImage(img.data)
del img_tif

# write all frames of h5 file as tiff files
for frame_id in range(0, img.nframes):
    frame_id_str = str(frame_id)
    frame_id_str = frame_id_str.rjust(no_digits, '0')
    image_full_name = root_file_name + separator + frame_id_str + ".tiff"
    path_to_save_multiple_tif_files = path_to_output_folder_multiple_tif_files + "/" + image_full_name
    img_tif = TiffIO(path_to_save_multiple_tif_files, mode='w')
    img_tif.writeImage(img.get_frame(frame_id).data)
    del img_tif
