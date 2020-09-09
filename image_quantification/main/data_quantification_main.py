# -*- coding: utf-8 -*-
# @Author: sf942274
# @Date:   2020-04-01 08:09:19
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-09 01:52:06

import io, os, sys, types, pickle, datetime, time, warnings

import pandas as pd

import numpy as np
from numpy.linalg import eig, inv

import math
from scipy import interpolate, spatial, stats

import skimage.io as skiIo
from skimage import exposure, img_as_float, filters

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse

import seaborn as sns


### ignore warnings
warnings.filterwarnings('ignore')

### include folders with additional functions
module_path = os.path.join(os.path.dirname(os.getcwd()), 'functions')
sys.path.insert(0, module_path)

import settings as settings
import helper as my_help
import intensity_calculation as my_int
import parse_bundle as my_pb
import plotting as my_plot

### import data
def import_data():

	"""
	Function: Import data.
	Input:
	Outputs:
	- roi_df: ROI csv as dataframe
	- annots_df_current: annotation csv as dataframe
	- image: image as matrix
	- image_info: list. contains: image_name, image_shape, m2p_ratio(um to pixel info)
	"""

	### params.
	paths = settings.paths

	### import annotation file
	annots_df = pd.read_csv(paths.annot_path, engine = 'python')
	image_list = annots_df.loc[:,'Image_Name'].unique()
	ROI_list = annots_df.loc[:,'ROI_Name'].unique()

	i_image = 0

	### import ROI csv file
	image_name = image_list[i_image]
	roi_name = ROI_list[i_image]
	roi_df = pd.read_csv(os.path.join(paths.roi_path, roi_name))
	roi_df.rename(columns = {' ':'No'}, inplace = True)
	annots_df_current = annots_df.groupby(['Image_Name']).get_group(image_list[i_image]).reset_index(drop = True)

	### import image matrix.
	image = img_as_float(skiIo.imread(os.path.join(paths.image_path, image_name)))
	image_shape = (image.shape[0], image.shape[1], image.shape[2])
	m2p_ratio = (annots_df.iloc[0]['imgX_pixel']/annots_df.iloc[0]['imgX_um'], annots_df.iloc[0]['imgY_pixel']/annots_df.iloc[0]['imgY_um']) # pixel/um for x and y axis.

	image_info = [image_name, image_shape, m2p_ratio]

	return roi_df, annots_df_current, image, image_info


### process annotation information.
def process_annotation(roi_df, annots_df_current, m2p_ratio):

	"""
	Function: Process annotation information.
	Inputs: roi_df, annots_df_current, m2p_ratio
	Outputs:
	- bundles_df: 
	- annots_df_current: updated annotation dataframe
	"""

	bundles_df = my_pb.get_bundles_info(roi_df, annots_df_current, m2p_ratio[0], m2p_ratio[1], is_print = False)
	annot_bundles_df = bundles_df.groupby('Bundle_Type').get_group('Heel')
	annot_bundles_df.dropna(axis=1, how='all', inplace = True)
	annot_bundles_df.fillna(value = 0, inplace = True)
	annot_bundles_df.sort_index(inplace = True)
	
	return bundles_df, annot_bundles_df

### process images: intensity normalization.
def process_image(image, image_shape):

	"""
	Function: Process images: intensity normalization.
	Inputs: image, image_shape
	outputs: img_norm in matrix form (normalized images)
	"""

	### params
	paths = settings.paths
	matching_info = settings.matching_info
	analysis_params_general = settings.analysis_params_general

	### number of channels
	nChannels = image.shape[3]
	num_norm_channels = len(settings.matching_info.channel_cmap.keys()) # number of channels of normalized image

	### normalize channels
	image_norm = np.empty(image_shape + (num_norm_channels,), dtype=image[:,:,:,1].dtype, order='C')

	### calculating R3/R4 density maps
	if(matching_info.channels_type == 'R3R4'):
		
		thr = np.zeros((2))

		# RFP_norm
		image_norm[:,:,:,0] = exposure.rescale_intensity(image[:,:,:,0], in_range = 'image', out_range='dtype')
		# GFP_norm
		image_norm[:,:,:,1] = exposure.rescale_intensity(image[:,:,:,1], in_range = 'image', out_range='dtype')    
		
		del image

	### checking raw images
	elif(matching_info.channels_type == 'checking'):
		image_norm = image
		del image

	else:
		print('ERROR! Please specify which channel type!\n')
		my_help.print_to_log("ERROR! Please specify which channel type!\n")
	
	return image_norm


### Convert image matrix to standardized density map matrix.
def analyze_image(bundles_df, annot_bundles_df, image_norm, image_name, m2p_ratio):

	"""
	Function: From raw image to normalized bundles.
	Inputs: 
	- bundles_df: DataFrame. Contains heels and targets info.
	- annot_bundles_df: DataFrame. Contains annotated heels info.
	- image_norm: numpy array. intensity normalized raw image.
	- image_name: string. name of the raw image.
	Outputs: 
	- intensity_matrix: numpy array. standardized density map matries for each channel and for each bundle.
	- params: list. list of parameters to passed on for intensity calculation function for each bundles 
	- rel_points: dictionary. {bundle_no : dictionary}. relative lengths for target positions and heel positions for each bundle.
	- thr_otsu: otsu threshold for each channels of the image.
	- thr_li: li threshold for each channels of the image.
	"""

	### parameters
	analysis_params_general = settings.analysis_params_general
	matching_info = settings.matching_info

	target_id_to_index = settings.matching_info.target_id_to_index
	slice_type = settings.analysis_params_general.slice_type

	### initialization
	print('-----' + image_name + '------')
	my_help.print_to_log(f'----- {image_name} ------\n')

	matrix_y = analysis_params_general.num_angle_section + 2 * analysis_params_general.num_outside_angle + 1
	matrix_x = analysis_params_general.num_x_section + 1
	matrix_z = analysis_params_general.z_offset * 2 + 1
	num_norm_channels = image_norm.shape[-1]

	ind_part = int(len(annot_bundles_df.index))
	bundles_list = annot_bundles_df.index
	print(f'Bundle Nos: {bundles_list.tolist()}')
	my_help.print_to_log(f'Bundle Nos: {bundles_list.tolist()}\n')
	
	intensity_matrix = np.zeros((len(bundles_list), num_norm_channels, matrix_y, matrix_x, matrix_z))
	intensity_matrix = intensity_matrix - 100
	
	params = [];
	rel_points = {}

	### thresholds
	print("Calculating thresholds...")
	my_help.print_to_log("Calculating thresholds...: ")
	thr_otsu = np.zeros((num_norm_channels))
	thr_li = np.zeros((num_norm_channels))
	time_start = time.time()
	for channel_no in range(num_norm_channels):
		thr_otsu[channel_no] = filters.threshold_otsu(image_norm[:,:,:,channel_no])
		thr_li[channel_no] = filters.threshold_li(image_norm[:,:,:,channel_no])
	time_end = time.time()
	time_dur = time_end - time_start
	print("total time: " + str(time_dur))
	my_help.print_to_log(f"total time: {time_dur}\n")

	### process
	for ind, bundle_no in enumerate(bundles_list):
		print(f'Bundle No {bundle_no}: ', end = " ")
		my_help.print_to_log(f'Bundle No {bundle_no}: ')

		### targets info
		ind_targets, coord_Tcs = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'c')
		ind_targets, coord_Te1s = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'e1')
		ind_targets, coord_Te2s = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'e2')
		# coord_center = my_help.get_bundle_center(bundle_no, bundles_df)
		coord_heels = my_help.get_heel_coords(bundle_no, bundles_df)

		### slice info
		## center of target ellipse as reference point
		slice_neg_one_point = coord_Tcs[target_id_to_index[7],:]
		slice_one_point = coord_Tcs[target_id_to_index[3],:]
		length_one_point = coord_Tcs[target_id_to_index[4],:]
		
		## center: intersection between R3-T3 and R4-T4
		cx, cy = my_help.line_intersection((coord_heels[2,:], slice_one_point),(coord_heels[3,:], slice_neg_one_point))
		center_point = np.array([cx, cy])
		
		### target range infow
		coord_Tls = np.zeros((len(coord_Tcs), 2)) # lower end
		coord_Ths = np.zeros((len(coord_Tcs), 2)) # higher end

		for i in range(len(coord_Tcs)):
			# coord_Tls[i,:] = my_int.get_relative_axis(coord_center[0,:], coord_Te1s[i,:], coord_Te2s[i,:], 'low')
			# coord_Ths[i,:] = my_int.get_relative_axis(coord_center[0,:], coord_Te1s[i,:], coord_Te2s[i,:], 'high')
			coord_Tls[i,:] = my_int.get_relative_axis(center_point, coord_Te1s[i,:], coord_Te2s[i,:], 'low')
			coord_Ths[i,:] = my_int.get_relative_axis(center_point, coord_Te1s[i,:], coord_Te2s[i,:], 'high')
		

		### get slicing params and calculate matrix
		bundle_params = [bundle_no, 
						ind_targets, 
						coord_Tcs,
						coord_Tls,
						coord_Ths,
						center_point,
						slice_neg_one_point, 
						slice_one_point, 
						length_one_point, 
						center_point,
						4]

		### calculate slicing information
		#### Angle normalization v3: T7 = -1, T4 = 0, T3 = 1
		if(slice_type == 0):
			pp_i, rel_points_i, fig  = my_int.get_slice_params_v1(bundles_df, bundle_params, image_name, xy_ratio = m2p_ratio[0], is_print = False, is_plot = True, is_save = True)

		#### Angle normalization v1: T7 = -1, T3 = 1
		elif(slice_type == 1):
			pp_i, rel_points_i, fig = my_int.get_slice_params_v2(bundles_df, bundle_params, image_name, xy_ratio = m2p_ratio[0], is_print = False, is_plot = True, is_save = True)

		plt.close(fig)
		params.append(pp_i)
		rel_points[ind] = rel_points_i

		#### calculate matrix
		time_start = time.time()
		for channel_no in range(num_norm_channels):
			print(f'ch_{channel_no},', end = " ")
			my_help.print_to_log(f'ch_{channel_no},')
			intensity_matrix[ind, channel_no,:,:,:] = my_int.get_intensity_matrix_new(pp_i, image_norm[:,:,:,channel_no])
		time_end = time.time()
		time_dur = time_end - time_start
		my_help.print_to_log(f'. Total time: {time_dur}\n' )

	return intensity_matrix, params, rel_points, thr_otsu, thr_li


### produce heat-map and polar-density plots.
def produce_figures(bundles_df, annot_bundles_df, intensity_matrix, params, rel_points, image_norm, img_name, thr_otsu, thr_li):
	
	"""
	Function: produce heat-map and polar-density plots.
	Inputs: 
	- bundles_df: DataFrame. Contains heels and targets info.
	- annot_bundles_df: DataFrame. Contains annotated heels info.
	- intensity_matrix: numpy array. standardized density map matrixs for each channel for each bundle.
	- params: list. list of parameters to passed on for intensity calculation function for each bundles 
	- rel_points: dictionary. {bundle_no : dictionary}. relative lengths for target positions and heel positions for each bundle.
	- image_norm: numpy array. intensity-scaled raw image.
	- image_name: string. name of the raw image.
	- thr_otsu: otsu threshold for each channels of the intensity-scaled raw image.
	- thr_li: li threshold for each channels of the intensity-scaled raw image.

	Outputs: 
	Figures saved in the appropriate output folder.
	"""


	### parameters
	paths = settings.paths
	analysis_params_general = settings.analysis_params_general
	matching_info = settings.matching_info
	num_norm_channels = image_norm.shape[-1]

	ind_part = int(len(annot_bundles_df.index))
	bundles_list = annot_bundles_df.index
	
	### loop through bundles.
	for ind, bundle_no in enumerate(bundles_list):
		print(f'Bundle No {bundle_no}:', end = " ")
		my_help.print_to_log(f'Bundle No: {bundle_no}: ')

		category_id = annot_bundles_df.iloc[0]['CategoryID']
		sample_id = annot_bundles_df.iloc[0]['SampleID']
		region_id = annot_bundles_df.iloc[0]['RegionID']

		#### parameters
		pp_i = params[ind]
		rel_points_i = rel_points[ind]

		#### density map
		matrix = my_help.delete_zero_columns(intensity_matrix[ind, :, :, :, :], -100, 3)


		if(len(matrix.flatten()) > 0):
			#### heat map
			print("HeatMap:", end = " ")
			my_help.print_to_log("HeatMap: ")

			time_start = time.time()

			plt.ioff()
			ori_x = np.round(np.linspace(0, analysis_params_general.radius_expanse_ratio, matrix.shape[2]), 2)

			thrs = np.zeros(6)
			thrs[2] = thr_otsu[0]
			thrs[3] = thr_otsu[1]
			thrs[4] = thr_li[0]
			thrs[5] = thr_li[1]
			thr_names = ['0', '0', 'Otsu', 'Otsu', 'Li', 'Li']
			fig_name = f'{category_id}_s{sample_id}r{region_id}_{matching_info.channels_type}_bundle_no_{bundle_no}.png'
			fig_params = [pp_i, img_name, fig_name]
			thr_params = [thrs, thr_names, 6]
			fig = my_plot.plot_bundle_vs_matrix(bundle_no, bundles_df, image_norm, matrix, 
														fig_params, thr_params, 
														is_label_off = True, is_save = True, 
														is_true_x_tick = True)
				
			time_end = time.time()
			time_dur = time_end - time_start

			print(f"total_time={time_dur}", end = " ")
			my_help.print_to_log(f"total_time={time_dur}")
			
			#### polar density map
			fig_params = [pp_i, img_name]
			print("; Polar: channels", end = " ")
			my_help.print_to_log("; Polar: channels")
			time_start = time.time()
			for channel_no in range(num_norm_channels):
				fig = my_plot.plot_polar(bundle_no, bundles_df, image_norm, channel_no, matrix, 
											fig_params, rel_points_i, 
											is_label_off = True, is_save = True, is_extended_target = True)
				plt.close(fig)
				print(f'{channel_no}-', end = " ")
				my_help.print_to_log(f'{channel_no}-')
			time_end = time.time()
			time_dur = time_end - time_start
			print(f"total_time={time_dur}")
			my_help.print_to_log(f"total_time={time_dur}\n")

		else:
			print("error! No intensity matrix calculated!")
			my_help.print_to_log("error! No intensity matrix calculated!")


### save results to a pickle file.
def save_results(annot_bundles_df, intensity_matrix, params, rel_points):

	"""
	Function: Save results.
	Inputs: 
	- annot_bundles_df
	- intensity_matrix
	- params
	- rel_points
	Outputs: output_data as adictionary saved in output folder
	- component of output_data: category_id, sample_id, region_id, slice_type, intensity_matrix, parameter, relative_positions
	"""

	analysis_params_general = settings.analysis_params_general
	paths = settings.paths
	matching_info = settings.matching_info
	# image_path, roi_path, annot_path, log_path, fig_out_prefix, data_out_prefix = paths
	category_id = annot_bundles_df.iloc[0]['CategoryID']
	time_id = annot_bundles_df.iloc[0]['TimeID']
	sample_id = annot_bundles_df.iloc[0]['SampleID']
	region_id = annot_bundles_df.iloc[0]['RegionID']
	channels_type = matching_info.channels_type
	ind_part = int(len(annot_bundles_df.index))
	bundles_list = annot_bundles_df.index

	output_data = {
		'category_ID': category_id,
		'time_ID':time_id,
		'sample_ID': sample_id,
		'region_ID': region_id,
		'channels_type': channels_type,
		'slice_type': analysis_params_general.slice_type,
		'intensity_matrix': intensity_matrix,
		'parameter': params,
		'relative_positions': rel_points,
		'bundle_nos': list(bundles_list),
		'annot_csv': paths.annot_name,
		'analysis_params_general':analysis_params_general
	}

	now = datetime.datetime.now()
	date_info = str(now.year)+str(now.month)+str(now.day)
	output_name = f'{category_id}_{time_id}hrs_sample{sample_id}_region{region_id}_slice{analysis_params_general.slice_type}_{channels_type}_v{date_info}.pickle'

	output_dir = os.path.join(paths.data_out_prefix)
	my_help.check_dir(output_dir)
	output_dir = os.path.join(output_dir,category_id)
	my_help.check_dir(output_dir)
	output_name = os.path.join(output_dir, output_name)
	pickle_out = open(output_name,"wb")
	pickle.dump(output_data, pickle_out)
	pickle_out.close()

	output_dir = os.path.join(paths.data_out_prefix)
	my_help.check_dir(output_dir)
	output_dir = os.path.join(output_dir,category_id)
	my_help.check_dir(output_dir)
	output_name = os.path.join(output_dir, output_name)
	pickle_out = open(output_name,"wb")
	pickle.dump(output_data, pickle_out)
	pickle_out.close()


"""
main function
"""
def main():
	analysis_params_general = settings.analysis_params_general
	paths = settings.paths
	matching_info = settings.matching_info

	print(f'================= {paths.annot_name} Analysis Start! =================')
	my_help.print_to_log(f'================= {paths.annot_name} Analysis Start! =================\n')

	roi_df, annots_df_current, image, image_info = import_data()		# image_info = [image_name, image_shape, m2p_ratio] 
	print("Data import finished!")
	my_help.print_to_log("Data import finished!\n")

	bundles_df, annot_bundles_df = process_annotation(roi_df, annots_df_current, image_info[2])
	print("annot_bundles_df done!")
	my_help.print_to_log("annot_bundles_df done!\n")

	time_start = time.time()
	image_norm = process_image(image, image_info[1])
	time_end = time.time()
	time_dur = time_end - time_start
	print("image processed! total time: ", time_dur)
	my_help.print_to_log(f'image processed! total time: {time_dur}\n')
	
	intensity_matrix, params, rel_points, thr_otsu, thr_li = analyze_image(bundles_df, annot_bundles_df, image_norm, image_info[0], image_info[2])
	print("image analyzed!")
	my_help.print_to_log("image analyzed!\n")

	save_results(annot_bundles_df,intensity_matrix, params, rel_points)
	print("data saved!")
	my_help.print_to_log("data saved!\n")
	
	produce_figures(bundles_df, annot_bundles_df, intensity_matrix, params, rel_points, image_norm, image_info[0], thr_otsu, thr_li)
	print("image results generated!")
	print("================= End of Analysis =================")
	my_help.print_to_log("image results generated!\n")
	my_help.print_to_log("================= End of Analysis =================")

if __name__ == "__main__":

	main()
