# -*- coding: utf-8 -*-
# @Author: sf942274
# @Date:   2020-04-01 08:09:19
# @Last Modified by:   sf942274
# @Last Modified time: 2020-08-09 18:22:48

import io, os, sys, types, pickle, datetime, time

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse

import numpy as np
from numpy.linalg import eig, inv
import math

from scipy import interpolate, spatial, stats

import seaborn as sns

import skimage.io as skiIo
from skimage import exposure, img_as_float, filters, morphology, transform

from sklearn import linear_model, metrics

### include folders with additional functions
# module_path = os.path.join(os.path.dirname(os.getcwd()), 'functions')
# module_path = '/wynton/home/awlab/lmorinishi/code/functions'
module_path = '/awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/functions'
sys.path.insert(0, module_path)

import settings as settings
import helper as my_help
import intensity_calculation_v4 as my_int
import parse_bundle as my_pb
import plotting_v4 as my_plot

"""
Function: Import data.
Input:
Outputs:
- roi_df: ROI csv as dataframe
- annot_df: annotation csv as dataframe
- image: image as matrix
- image_info: list. contains: image_name, image_shape, m2p_ratio(um to pixel info)
"""
def import_data():
	paths = settings.paths

	summary_df = pd.read_csv(paths.annot_path)
	image_list = summary_df.loc[:,'Image_Name'].unique()
	ROI_list = summary_df.loc[:,'ROI_Name'].unique()

	i_image = 0

	image_name = image_list[i_image]
	roi_name = ROI_list[i_image]
	roi_df = pd.read_csv(os.path.join(paths.roi_path, roi_name))
	roi_df.rename(columns = {' ':'No'}, inplace = True)
	annot_df = summary_df.groupby(['Image_Name']).get_group(image_list[i_image]).reset_index(drop = True)

	image = img_as_float(skiIo.imread(os.path.join(paths.image_path, image_name)))
	image_shape = (image.shape[0], image.shape[1], image.shape[2])
	m2p_ratio = (summary_df.iloc[0]['imgX_pixel']/summary_df.iloc[0]['imgX_um'], summary_df.iloc[0]['imgY_pixel']/summary_df.iloc[0]['imgY_um'])

	image_info = [image_name, image_shape, m2p_ratio]

	return roi_df, annot_df, image, image_info


"""
Function: Process annotation information.
Inputs: roi_df, annot_df, m2p_ratio
Outputs:
- bundles_df: 
- annot_df: updated annotation dataframe
"""
def process_annotation(roi_df, annot_df, m2p_ratio):
	is_extended_target_list = False
	annotation_type = annot_df.loc[0,'Annotation_type']
	if(annotation_type != 4):
		if(annotation_type == 1): #old annotation: bundle center at last
			bundles_df = my_pb.get_bundles_info_v1(roi_df, annot_df, m2p_ratio[0], m2p_ratio[1], is_extended_target_list)
		elif(annotation_type == 2): #new annotation: T0 at last 
			bundles_df = my_pb.get_bundles_info_v2(roi_df, annot_df, m2p_ratio[0], m2p_ratio[1], is_extended_target_list)
		elif(annotation_type == 3):
			bundles_df = my_pb.get_bundles_info_v3(roi_df, annot_df, m2p_ratio[0], m2p_ratio[1])
		
		annot_bundles_df = bundles_df.dropna(axis=0, how='any', inplace = False)

		annot_bundles_df.sort_index(inplace = True)

	elif(annotation_type == 4):
		bundles_df = my_pb.get_bundles_info_v4(roi_df, annot_df, m2p_ratio[0], m2p_ratio[1], is_print = False)
		annot_bundles_df = bundles_df.groupby('Bundle_Type').get_group('Heel')
		annot_bundles_df.dropna(axis=1, how='all', inplace = True)
		annot_bundles_df.fillna(value = 0, inplace = True)
		annot_bundles_df.sort_index(inplace = True)
	
	return bundles_df, annot_bundles_df


"""
Function: Process images: intensity normalization, r3_img/r4_img channel extraction.
Inputs: image, image_shape
Outputs: img_norm in matrix form (normalized images)
"""
def process_image(image, image_shape):
	paths = settings.paths
	matching_info = settings.matching_info
	analysis_params_general = settings.analysis_params_general

	### number of channels
	nChannels = image.shape[3]
	num_norm_channels = len(settings.matching_info.channel_cmap.keys()) # number of channels of normalized image

	### normalize channels
	image_norm = np.empty(image_shape + (num_norm_channels,), dtype=image[:,:,:,1].dtype, order='C')
	# image_norm = np.random.randn(image_shape + (num_norm_channels,))
	if(matching_info.channels_type == 'R3R4'):
		
		thr = np.zeros((2))

		# RFP_norm
		image_norm[:,:,:,0] = exposure.rescale_intensity(image[:,:,:,0], in_range = 'image', out_range='dtype')
		# GFP_norm
		image_norm[:,:,:,1] = exposure.rescale_intensity(image[:,:,:,1], in_range = 'image', out_range='dtype')    
		
		del image
		
		print("gfp threshold!")
		my_help.print_to_log("gfp threshold!\n")
		thr[0] = filters.threshold_isodata(image_norm[:,:,:,1])
		thr[1] = filters.threshold_mean(image_norm[:,:,:,1])

		print("histogram matching!")
		my_help.print_to_log("histogram matching!\n")
		gfp = transform.match_histograms(image_norm[:,:,:,1], image_norm[:,:,:,0])
		
		print("R3/R4 v1")
		my_help.print_to_log("R3/R4 v1\n")
		r3_img = image_norm[:,:,:,0] - gfp
		r3_img[r3_img<0] = 0
		image_norm[:,:,:,2] = exposure.rescale_intensity(r3_img, in_range = 'image', out_range='dtype')
		r4_img = image_norm[:,:,:,0] * gfp
		image_norm[:,:,:,3] = exposure.rescale_intensity(r4_img, in_range = 'image', out_range='dtype')
		image_norm[:,:,:,2] = image_norm[:,:,:,0]
		image_norm[:,:,:,3] = image_norm[:,:,:,1]
		
		print("R3/R4 v2")
		my_help.print_to_log("R3/R4 v2\n")
		gfp_thr = morphology.binary_opening((image_norm[:,:,:,1]>thr[0])*1)
		image_norm[:,:,:,4] = exposure.rescale_intensity(image_norm[:,:,:,0] * (1-gfp_thr), in_range = 'image', out_range='dtype')
		image_norm[:,:,:,5] = exposure.rescale_intensity(morphology.closing(image_norm[:,:,:,1]*((image_norm[:,:,:,1]>((thr[0] + thr[1])/2))*1)))
		image_norm[:,:,:,4] = image_norm[:,:,:,0]
		image_norm[:,:,:,5] = image_norm[:,:,:,1]

		print("R3 v3")
		my_help.print_to_log("R3 v3\n")
		r3_img = image_norm[:,:,:,0] - gfp*settings.analysis_params_general.scale_factor
		r3_img[r3_img<0] = 0
		image_norm[:,:,:,6] = exposure.rescale_intensity(r3_img, in_range = 'image', out_range='dtype')
		image_norm[:,:,:,6] = image_norm[:,:,:,0]

		del r3_img, r4_img, gfp, gfp_thr

	elif(matching_info.channels_type == 'FasII'):
		# RFP_norm
		image_norm[:,:,:,0] = exposure.rescale_intensity(image[:,:,:,0], in_range = 'image', out_range='dtype')
		# GFP_norm
		image_norm[:,:,:,1] = exposure.rescale_intensity(image[:,:,:,1], in_range = 'image', out_range='dtype')    
		# FasII_norm
		image_norm[:,:,:,2] = exposure.rescale_intensity(image[:,:,:,3], in_range = 'image', out_range='dtype')
		
		del image

		print("fasII threshold!\n")
		thr = filters.threshold_otsu(image_norm[:,:,:,2])

		print("FasII intersection!\n")
		fasii_thr = morphology.binary_opening((image_norm[:,:,:,2]>thr)*1)
		image_norm[:,:,:,3] = exposure.rescale_intensity(image_norm[:,:,:,0] * fasii_thr, in_range = 'image', out_range='dtype')
		image_norm[:,:,:,4] = exposure.rescale_intensity(image_norm[:,:,:,1] * fasii_thr, in_range = 'image', out_range='dtype')
		
		del fasii_thr

	else:
		print('ERROR! Please specify which channel type!\n')
		my_help.print_to_log("ERROR! Please specify which channel type!\n")
	
	return image_norm


"""
Function: From raw image to normalized bundles.
Inputs: bundles_df, annot_bundles_df, image_norm, image_name
Outputs: intensity_matrix, params, rel_points, thr_otsu, thr_li, thr_isodata
"""
def analyze_image(bundles_df, annot_bundles_df, image_norm, image_name, m2p_ratio):
	### parameters
	analysis_params_general = settings.analysis_params_general
	matching_info = settings.matching_info

	target_id_to_index = settings.matching_info.target_id_to_index
	center_type = settings.analysis_params_general.center_type
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
	thr_isodata = np.zeros((num_norm_channels))
	time_start = time.time()
	for channel_no in range(num_norm_channels):
		thr_otsu[channel_no] = filters.threshold_otsu(image_norm[:,:,:,channel_no])
		thr_li[channel_no] = filters.threshold_li(image_norm[:,:,:,channel_no])
		# thr_isodata[channel_no] = filters.threshold_isodata(image_norm[:,:,:,channel_no])
	time_end = time.time()
	time_dur = time_end - time_start
	print("total time: " + str(time_dur))
	my_help.print_to_log(f"total time: {time_dur}\n")

	### process
	for ind, bundle_no in enumerate(bundles_list):
		print(f'Bundle No {bundle_no}: ', end = " ")
		my_help.print_to_log(f'Bundle No {bundle_no}: ')

		### targets info
		ind_targets, coord_Tcs = my_help.get_target_v4(bundle_no, bundles_df, return_type = 'c')
		ind_targets, coord_Te1s = my_help.get_target_v4(bundle_no, bundles_df, return_type = 'e1')
		ind_targets, coord_Te2s = my_help.get_target_v4(bundle_no, bundles_df, return_type = 'e2')
		coord_center = my_help.get_bundle_center(bundle_no, bundles_df)
		coord_heels = my_help.get_heel_coords(bundle_no, bundles_df)

		### slice info
		## center of target ellipse as reference point
		slice_neg_one_point = coord_Tcs[target_id_to_index[7],:]
		slice_one_point = coord_Tcs[target_id_to_index[3],:]
		length_one_point = coord_Tcs[target_id_to_index[4],:]
		
		## center: intersection between R3-T3 and R4-T4
		cx, cy = my_help.line_intersection((coord_heels[2,:], slice_one_point),(coord_heels[3,:], slice_neg_one_point))
		center_points = [coord_center[0,:], np.array([cx, cy])]
		
		### target range infow
		coord_Tls = np.zeros((len(coord_Tcs), 2)) # lower end
		coord_Ths = np.zeros((len(coord_Tcs), 2)) # higher end

		for i in range(len(coord_Tcs)):
			coord_Tls[i,:] = my_int.get_relative_axis(coord_center[0,:], coord_Te1s[i,:], coord_Te2s[i,:], 'low')
			coord_Ths[i,:] = my_int.get_relative_axis(coord_center[0,:], coord_Te1s[i,:], coord_Te2s[i,:], 'high')
		
		r_cell_nos = [4,4]

		### get slicing params and calculate matrix
		bundle_params = [bundle_no, 
						ind_targets, 
						coord_Tcs,
						coord_Tls,
						coord_Ths,
						center_points[center_type],
						slice_neg_one_point, 
						slice_one_point, 
						length_one_point, 
						center_points[center_type],
						r_cell_nos[center_type]]
		if(slice_type == 0):
			pp_i, rel_points_i, fig  = my_int.get_slice_params_v1(bundles_df, bundle_params, image_name, xy_ratio = m2p_ratio[0], is_print = False, is_plot = True, is_save = True)
		elif(slice_type == 1):
			pp_i, rel_points_i, fig = my_int.get_slice_params_v3(bundles_df, bundle_params, image_name, xy_ratio = m2p_ratio[0], is_print = False, is_plot = True, is_save = True)
		plt.close(fig)
		params.append(pp_i)
		rel_points[ind] = rel_points_i

		# calculate matrix
		time_start = time.time()
		for channel_no in range(num_norm_channels):
		# for channel_no in [0]:
			print(f'{channel_no},', end = " ")
			my_help.print_to_log(f'{channel_no}, ')
			intensity_matrix[ind, channel_no,:,:,:] = my_int.get_intensity_matrix_new(pp_i, image_norm[:,:,:,channel_no])
			# intensity_matrix[ind, channel_no,:,:,:] = np.random.randn(intensity_matrix[ind, channel_no,:,:,:].shape[0], intensity_matrix[ind, channel_no,:,:,:].shape[1], intensity_matrix[ind, channel_no,:,:,:].shape[2])
		time_end = time.time()
		time_dur = time_end - time_start
		my_help.print_to_log(f'. Total time: {time_dur}' )

	return intensity_matrix, params, rel_points, thr_otsu, thr_li, thr_isodata


"""
Function: Make figures.
Inputs: bundles_df, annot_bundles_df, intensity_matrix, params, rel_points, image_norm, img_name, thr_otsu, thr_li, thr_isodata
Outputs: figures saved in output folders
"""
def produce_figures(bundles_df, annot_bundles_df, intensity_matrix, params, rel_points, image_norm, img_name, thr_otsu, thr_li, thr_isodata):
	### parameters
	paths = settings.paths
	analysis_params_general = settings.analysis_params_general
	matching_info = settings.matching_info
	num_norm_channels = image_norm.shape[-1]

	ind_part = int(len(annot_bundles_df.index))
	bundles_list = annot_bundles_df.index
	
	for ind, bundle_no in enumerate(bundles_list):
		print(f'Bundle No {bundle_no}:', end = " ")
		my_help.print_to_log(f'Bundle No: {bundle_no}: ')

		category_id = annot_bundles_df.iloc[0]['CategoryID']
		sample_id = annot_bundles_df.iloc[0]['SampleID']
		region_id = annot_bundles_df.iloc[0]['RegionID']

		### parameters
		pp_i = params[ind]
		rel_points_i = rel_points[ind]

		matrix = my_help.delete_zero_columns(intensity_matrix[ind, :, :, :, :], -100, 3)
		# if(sum(np.all(intensity_matrix[ind, :, :, :, :] == -100, axis = 0)[2].flatten()) == 0):
		if(len(matrix.flatten()) > 0):
			## heat map
			print("HeatMap:", end = " ")
			my_help.print_to_log("HeatMap: ")

			time_start = time.time()

			plt.ioff()
			ori_x = np.round(np.linspace(0, analysis_params_general.radius_expanse_ratio[analysis_params_general.center_type], matrix.shape[2]), 2)
			tick_params = [2, 1, ori_x, 21] ### tickTypeX, tickTypeY, tickArg2_X, tickArg2_Y

			## gfp and rfp map
			thrs = np.zeros(6)
			thrs[2] = thr_otsu[0]
			thrs[3] = thr_otsu[1]
			thrs[4] = thr_li[0]
			thrs[5] = thr_li[1]
			thr_names = ['0', '0', 'Otsu', 'Otsu', 'Li', 'Li']
			fig_name = f'{category_id}_s{sample_id}r{region_id}_{matching_info.channels_type}_bundle_no_{bundle_no}_GFPRFP.png'
			fig_params = [pp_i, img_name, fig_name]
			plot_options = [thrs, thr_names, 6]
			fig = my_plot.plot_bundle_vs_matrix_gr_v4(bundle_no, bundles_df, image_norm, matrix, 
														fig_params, tick_params, plot_options, 
														is_label_off = True, is_save = True, 
														is_true_x_tick = True, is_ori_tick = False)

			### other channels map
			for thr_function_ids in [0, 1, 2]: # different thresholding methods
				print(f'thr{thr_function_ids}:', end = " ")
				my_help.print_to_log(f'thr{thr_function_ids}: ')
				
				thrs = np.zeros((num_norm_channels))
				if(thr_function_ids == 0):
					thrs = np.zeros((num_norm_channels))
				elif(thr_function_ids == 1):
					thrs = thr_otsu
				elif(thr_function_ids == 2):
					thrs = thr_li
				# elif(thr_function_ids == 3):
				#   thrs = thr_isodata

				fig_name = f'{category_id}_s{sample_id}r{region_id}_{matching_info.channels_type}_bundle_no_{bundle_no}_{thr_function_ids}.png'
				fig_params = [pp_i, img_name, fig_name]
				plot_options = [thrs, thr_function_ids, num_norm_channels - 2]
				# fig = my_plot.plot_bundle_vs_matrix_all(bundle_no, bundles_df, image_norm, matrix, fig_params, tick_params, plot_options, is_label_off = True, is_save = True, is_true_x_tick = True, is_ori_tick = False)
				fig = my_plot.plot_bundle_vs_matrix_others_v4(bundle_no, bundles_df, image_norm, matrix, 
												fig_params, tick_params, plot_options, 
												is_label_off = True, is_save = True, 
												is_true_x_tick = True, is_ori_tick = False)
				plt.close(fig)		
				
			time_end = time.time()
			time_dur = time_end - time_start

			print(f"total_time={time_dur}", end = " ")
			my_help.print_to_log(f"total_time={time_dur}")

			
			# polar plot
			fig_params = [pp_i, img_name]
			print("; Polar: channels", end = " ")
			my_help.print_to_log("; Polar: channels")
			time_start = time.time()
			for channel_no in range(num_norm_channels):
				fig = my_plot.plot_polar_v4(bundle_no, bundles_df, image_norm, channel_no, matrix, 
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

"""
Function: Save results.
Inputs: annot_bundles_df, intensity_matrix, params, rel_points
Outputs: output_data saved in output folder
- component of output_data: category_id, sample_id, region_id, slice_type, center_type, intensity_matrix, parameter, relative_positions
"""
def save_results(annot_bundles_df, intensity_matrix, params, rel_points):
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
		'center_type': analysis_params_general.center_type,
		'intensity_matrix': intensity_matrix,
		'parameter': params,
		'relativePositions': rel_points,
		'bundle_nos': list(bundles_list),
		'annot_csv': paths.annot_name,
		'analysis_params_general':analysis_params_general
	}

	now = datetime.datetime.now()
	date_info = str(now.year)+str(now.month)+str(now.day)
	output_name = f'{category_id}_{time_id}hrs_sample{sample_id}_region{region_id}_slice{analysis_params_general.slice_type}_center{analysis_params_general.center_type}_{channels_type}_v{date_info}_01.pickle'

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

	roi_df, annot_df, image, image_info = import_data()		# image_info = [image_name, image_shape, m2p_ratio] 
	print("Data import finished!")
	my_help.print_to_log("Data import finished!\n")

	bundles_df, annot_bundles_df = process_annotation(roi_df, annot_df, image_info[2])
	print("annot_bundles_df done!")
	my_help.print_to_log("annot_bundles_df done!\n")

	time_start = time.time()
	image_norm = process_image(image, image_info[1])
	time_end = time.time()
	time_dur = time_end - time_start
	print("image processed! total time: ", time_dur)
	my_help.print_to_log(f'image processed! total time: {time_dur}\n')
	
	intensity_matrix, params, rel_points, thr_otsu, thr_li, thr_isodata = analyze_image(bundles_df, annot_bundles_df, image_norm, image_info[0], image_info[2])
	print("image analyzed!")
	my_help.print_to_log("image analyzed!\n")

	save_results(annot_bundles_df,intensity_matrix, params, rel_points)
	print("data saved!")
	my_help.print_to_log("data saved!\n")
	
	produce_figures(bundles_df, annot_bundles_df, intensity_matrix, params, rel_points, image_norm, image_info[0], thr_otsu, thr_li, thr_isodata)
	print("image results generated!")
	print("================= End of Analysis =================")
	my_help.print_to_log("image results generated!\n")
	my_help.print_to_log("================= End of Analysis =================")

if __name__ == "__main__":

	main()