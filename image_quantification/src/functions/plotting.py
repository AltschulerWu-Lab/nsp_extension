# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2021-12-04 18:04:19


import os, math, matplotlib

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
import seaborn as sns

import numpy as np
import numpy.ma as ma

from skimage import exposure


import helper as my_help
import intensity_calculation as my_int
import settings as settings


""" ========================= Supporting Functions =========================== """
### return coordinates of a circle given it's center and radius.
def draw_circle(center, radius):
	"""
	Function: return coordinates of a circle given it's center and radius
	Inputs:
	- center: numpy array (or list). x and y coordinates of the center of the circle.
	- radius: value of the radius of the circle.
	Output: numpy array. x and y coordinates of the circle
	"""
	theta = np.linspace(0, 2*np.pi, 100)

	# compute x1 and x2
	x = radius*np.cos(theta) + center[0]
	y = radius*np.sin(theta) + center[1]
	circle = np.column_stack((x,y))
	return circle

### convert from cartesian coordinate to polar coordinate
def cartesian_to_polar(x,y):
	"""
	Function: convert from cartesian coordinate to polar coordinate
	Input: x,y - float. x and y coordinate values.
	Output: r, theta - float. radial and angle coordinate values.
	"""
	r = np.sqrt(x**2 + y**2)
	theta = np.arctan(y/x)
	return r,theta

### convert from polar coordinate to cartesian coordinate
def polar_to_cartesian(r,theta):
	"""
	Function: convert from polar coordinate to cartesian coordinate
	Input: r, theta - float. radial and angle coordinate values.
	Output: x,y - float. x and y coordinate values. 
	"""
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	return x,y

### reduce x and y axis labels.
def get_tick_list(tick_type, ori_tick, arg2):
	"""
	Function: reduce axis labels.
	Inputs:
	- tick_type: int. 
		- tick_type == 1: specify number of labels remained.
		- tick_type == 2: specify which label should be kept.
	- ori_tick: numpy array. original tick values.
	- arg2: 
		- if tick_type == 1: int. number of labels
		- if tick_type == 2: list. labels to keep.
	Output: ticks_list: list. tick values.
	"""

	### specify number of labels remained.
	if(tick_type == 1):
		num_of_labels = arg2
		k = np.ceil((len(ori_tick) - 1)/(num_of_labels)).astype(int)
		nk = np.floor(len(ori_tick)/k).astype(int)

		true_label_list = []
		i = 0
		ti = 0 + i*k
		while ti < len(ori_tick):
			true_label_list.append(0 + i*k)
			i += 1
			ti = 0 + i*k
	
	### specify which label should be kept.
	elif(tick_type == 2):
		must_lists = arg2
		true_label_list = np.where(np.isin(ori_tick, must_lists))[0]
	
	### from index of ticks to tick values
	tick_list = []
	for i in range(len(ori_tick)):
		if i in true_label_list:
			tick_list.append(ori_tick[i].astype(str))
		else:
			tick_list.append('')

	return tick_list

### get slicing information for polar plot.
def get_slice_params_for_polar_plot(analysis_params, slicing_params):
	"""
	Function: get slicing information for polar plot.
	Inputs:
	- analysis_params: list. Contains: num_angle_section, num_outside_angle, num_x_section, z_offset, radius_expanse_ratio.
	- slicing_params: list. Contains: slice_zero_point, slice_one_point, cut_off_point, center_point.
	Outputs:
	- rs: numpy array. coordinates of angular axis.
	- phis: numpy array. coordinates of radial axis.
	"""
	num_angle_section, num_outside_angle, num_x_section, z_offset, radius_expanse_ratio = analysis_params
	slice_zero_point, slice_one_point, cut_off_point, center_point = slicing_params
	
	radius = np.linalg.norm( center_point - cut_off_point )
	angle_start_to_r = np.arctan2( slice_zero_point[1] - center_point[1], slice_zero_point[0] - center_point[0] )
	angle_end_to_r = np.arctan2( slice_one_point[1] - center_point[1], slice_one_point[0] - center_point[0])
	
	phi_range = my_int.inner_angle(slice_one_point - center_point, slice_zero_point - center_point, True)
	phi_unit = phi_range/num_angle_section
	
	if(((-np.pi <= angle_start_to_r <= -0.5*np.pi) | (-np.pi <= angle_end_to_r <= -0.5*np.pi)) & (angle_start_to_r*angle_end_to_r < 1) ):
		if((-np.pi <= angle_start_to_r <= -0.5*np.pi) & (-np.pi <= angle_end_to_r <= -0.5*np.pi)):
			phi_start = min(angle_start_to_r, angle_end_to_r) - num_outside_angle * phi_unit
			phi_end = max(angle_start_to_r, angle_end_to_r) + num_outside_angle * phi_unit
		else:
			phi_start = max(angle_start_to_r, angle_end_to_r) - num_outside_angle * phi_unit
			phi_end = min(angle_start_to_r, angle_end_to_r) + num_outside_angle * phi_unit
	else:
		phi_start = min(angle_start_to_r, angle_end_to_r) - num_outside_angle * phi_unit
		phi_end = max(angle_start_to_r, angle_end_to_r) + num_outside_angle * phi_unit

	phi_start = my_int.angle_normalization(phi_start)
	phi_end = my_int.angle_normalization(phi_end)

	phis = my_int.get_phis(phi_start, phi_end, num_angle_section + num_outside_angle*2 + 2)

	if(my_int.smallest_angle(angle_start_to_r, phis[-1]) < my_int.smallest_angle(angle_start_to_r, phis[0])):
		phis = np.flip(phis, axis = 0)
	
	rs = np.linspace(0, radius_expanse_ratio, num_x_section + 2)
	
	return rs, phis

### get polar or cartasian coordinates of targets
def get_target_grid(return_type, **kwargs):
	"""
	Function: get polar or cartasian coordinates of targets
	Inputs:
	- return_type: str. "cart" for cartasian coordinates; "polar" for polar coordinates.
	- kwargs: additional params.
		- rel_points: dictionary. relative length for target positions and heel positions
	Outputs:
	- if return cartasian coordinates: numpy array. x and y coordinates of targets in cartasian coordinates.
	- if return polar coordinates: dictionary {type('c', 'l', 'h'):numpy array}. polar coordinates of target centers ('c')/lower bounds ('l')/upper bounds ('h')
	"""

	### unravel params.
	if('rel_points' in kwargs.keys()):
		rel_points = kwargs['rel_points']

	### calculate ideal grid
	#### before standardization
	##### distance: normal
	dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
	dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
	dT2T5 = dT3T7 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
	dT0T3 = dT0T7 = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5

	##### angles: normal
	aT0T2 = math.radians(80)/2
	aT0T5 = - math.radians(80)/2
	aT0T3 = math.acos((dT0T3 ** 2 + dT0T7 ** 2 - dT3T7 ** 2)/(2*dT0T3*dT0T7))/2
	aT0T7 = - aT0T3
	aT0T4 = 0

	##### target coordinates
	T0 = np.array((0,0))
	T2 = np.array((aT0T2, dT0T2))
	T3 = np.array((aT0T3, dT0T3))
	T4 = np.array((aT0T4, dT0T4))
	T5 = np.array((aT0T5, dT0T2))
	T7 = np.array((aT0T7, dT0T7))

	target_grid_polar = np.stack((T0, T2, T3, T4, T5, T7), axis = 0)
	target_grid_cart = np.zeros((6,2))
	for i in range(6):
		target_grid_cart[i,:] = polar_to_cartesian(target_grid_polar[i,1], target_grid_polar[i,0])

	##### heel coordinates
	alpha = 0.2354
	a = 0.2957
	b = 0.5
	r_heels_cart = np.zeros((6,2))
	r_heels_polar = np.zeros((6,2))
	for n in range(1,7):
		phi_n = -(alpha + (n-1)*(np.pi - 2*alpha)/5)
		x = a*np.cos(phi_n)
		y = b*np.sin(phi_n)
		r, theta = cartesian_to_polar(-y, x)
		r_heels_cart[n-1, :] = [-y,x]
		r_heels_polar[n-1, :] = [theta, r]

	##### intersect
	c = my_help.line_intersection((r_heels_cart[2,:], target_grid_cart[2,:]),(r_heels_cart[3,:], target_grid_cart[5,:]))

	#### after standardization
	dTiC = np.zeros((6,1))
	for i in range(1,6):
		dTiC[i] = np.linalg.norm(target_grid_cart[i,:] - c)
	dTiC = dTiC/dTiC[3]
	aTiCT4 = np.zeros((6,1))
	for i in range(1,6):
		aTiCT4[i] = my_int.inner_angle(target_grid_cart[i,:] - c, target_grid_cart[3,:] - c, True)
		if(i in [4,5]):
			aTiCT4[i] = - aTiCT4[i]

	### calculate output values
	if(return_type == 'cart'):
		grid_cart = np.zeros((6,2))
		for i in range(1,6):
			grid_cart[i,0],grid_cart[i,1] = polar_to_cartesian(dTiC[i][0], aTiCT4[i][0])
		
		return grid_cart

	elif(return_type == 'polar'):
		target_grid_polar = {}
		for t in ['c', 'l', 'h']:
			T0 = np.array((aTiCT4[0], -rel_points[f'T0{t}']))
			T2 = np.array((aTiCT4[1], rel_points[f'T2{t}']))
			T3 = np.array((aTiCT4[2], rel_points[f'T3{t}']))
			T4 = np.array((aTiCT4[3], rel_points[f'T4{t}']))
			T5 = np.array((aTiCT4[4], rel_points[f'T5{t}']))
			T3_ = np.array((aTiCT4[5], rel_points[f'T7{t}']))
			C0 = np.array((aTiCT4[0], rel_points['center']))
			target_grid_polar[t] = np.stack((T0, T2, T3, T4, T5, T3_, C0), axis = 0)

		return target_grid_polar


### get values for polar density plot
def get_polar_plot_values(analysis_params, channel_no, matrix, cmap, rel_points):

	"""
	Function: get values for polar density plot
	Inputs:
	- analysis_params: list.
	- channel_no: int. number of channel to plot, RFP (0) and GFP (1)
	- matrix: numpy array. mean(or max) standardized density map matrix for channel-of-interest and for bundle-of-interest.
	- cmap: plt.cmap. colormap used to plot this channel.
	- rel_points: dictionary. relative length for target positions and heel positions
	Outputs:
	- thetav, rv: numpy array. polar coordinate values of slicing grid.
	- z: transposed density map matrix
	- norm: output of BoundaryNorm function. heat-map normalization values.
	- target_grid_polar: dictionary {type('c', 'l', 'h'):numpy array}. polar coordinates of target centers ('c')/lower bounds ('l')/upper bounds ('h')
	"""

	### unravel parameters
	matching_info = settings.matching_info
	analysis_params_general = settings.analysis_params_general
	radius_expanse_ratio = analysis_params_general.radius_expanse_ratio

	### target grid in cartesian and polar coordinates.
	target_grid_polar = get_target_grid('polar', rel_points = rel_points)
	target_grid_cart = get_target_grid('cart')

	### calculate polar coordinate values of slicing grid.
	cut_off_point = 1 * analysis_params_general.radius_expanse_ratio
	bundle_pardams = target_grid_cart[matching_info.target_id_to_index[7]], target_grid_cart[matching_info.target_id_to_index[3]], cut_off_point, target_grid_cart[matching_info.target_id_to_index[0]] 

	rs, phis = get_slice_params_for_polar_plot(analysis_params, bundle_pardams)
	thetav, rv = np.meshgrid(phis, rs)
	
	### density map information.
	z = matrix.transpose()
	levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	return thetav, rv, z, norm, target_grid_polar

### get values to plot raw images.
def get_imshow_values(bundles_df, bundle_no, image, r_z, z_offset, channel_no):
	
	"""
	Function: get values to plot raw images
	Inputs:
	- bundle_no: int. Bundle No. for bundle-of-interest.
	- bundles_df: DataFrame. Contains heels and targets info.
	- image: numpy array. intensity-scaled raw image matrix.
	- r_z: int. z slice number for center of bundle image matrix.
	- z_offset: int. range of z slices (in both directions)
	- channel_no: int. number of channel to plot, RFP (0) and GFP (1)
	Outputs:
	- r_heel_coords: numpy array. heel coordinates.
	- coord_Tcs: numpy array. target coordinates.
	- target_ellipses: numpy array. params of target ellipses
	- bundle_img: numpy array. 2D image of the bundle.
	- xlim: list. [min, max] of x axis.
	- ylim: list. [min, max] of y axis.
	"""

	### params
	matching_info = settings.matching_info

	### R heels info
	r_heel_coords = my_help.get_heel_coords(bundle_no, bundles_df)

	### targets info
	ind_targets, coord_Tcs = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'c')
	_, coord_Te1s = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'e1')
	_, coord_Te2s = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'e2')
	_, target_ellipses = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'ellipse')
	coords = np.concatenate((coord_Tcs, coord_Te1s, coord_Te2s))

	### image boundary
	image_bd = (image.shape[2], image.shape[1])
	xlim = [max(np.floor(coords.min(axis = 0))[0] - 21, 0), min(np.ceil(coords.max(axis = 0))[0] + 21, image_bd[0])]
	ylim = [max(np.floor(coords.min(axis = 0))[1] - 21, 0), min(np.ceil(coords.max(axis = 0))[1] + 21, image_bd[1])]

	img_crop = image[r_z-z_offset : r_z+z_offset+1, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), channel_no]
	bundle_img = exposure.rescale_intensity(np.max(img_crop, axis = 0), in_range = 'image', out_range='dtype')

	return r_heel_coords, coord_Tcs, target_ellipses, bundle_img, xlim, ylim



""" ========================= Plotting =========================== """
### plot angle slicing information.
def plot_angles(phis, phi_edges, img_name, bundle_no, **kwargs):
	"""
	Function: plot angle slicing information.
	Input: 
	- phi_edges: list. two(or three) angles in radians
	- phis: np.ndarray. array of angles that should start from "start" and end at "end"
	- img_name: string. name of image being processed
	- bundle_no: int. Bundle No. for bundle-of-interest.
	Output: figure
	"""

	### params
	paths = settings.paths
	matching_info = settings.matching_info
	analysis_params_general = settings.analysis_params_general

	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False

	number = len(phis)
	cmap = plt.get_cmap('PuBu')
	colors = [cmap(i) for i in np.linspace(0, 1, number)]


	### figure
	fig = plt.figure(figsize = (6,6))
	ax = fig.add_subplot(111, projection='polar')
	sns.set_style("white")

	## set-up font sizes
	SMALL_SIZE = 12
	MEDIUM_SIZE = 14
	BIGGER_SIZE = 16
	plt.rc('font', size=SMALL_SIZE)         # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)   # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)   # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title

	for i, color in enumerate(colors, start=0):
		ax.plot(phis[i], 1, 'o', color = color)
	ax.plot([0, phis[0]], [0,1], '--', linewidth = 4.0, color = matching_info.color_code[7])
	ax.plot([0, phis[-1]], [0,1], '--', linewidth = 4.0, color = matching_info.color_code[3])
	if(len(phi_edges) == 2):
		ax.plot([0,phi_edges[0]], [0,1], '-', linewidth = 4.0, color = matching_info.color_code[7])
		ax.plot([0,phi_edges[1]], [0,1], '-', linewidth = 4.0, color = matching_info.color_code[3])
	elif(len(phi_edges) == 3):
		ax.plot([0,phi_edges[0]], [0,1], '-', linewidth = 4.0, color = matching_info.color_code[7])
		ax.plot([0,phi_edges[1]], [0,1], '-', linewidth = 4.0, color = matching_info.color_code[4])
		ax.plot([0,phi_edges[2]], [0,1], '-', linewidth = 4.0, color = matching_info.color_code[3])
	
	plt.title(f'Bundle_No {bundle_no}: s{analysis_params_general.slice_type}')

	if(is_save):
		plt.ioff()
		folder_name = img_name
		subfolder1 = 'Angle_Plot'
		subfolder2 = f'slice_type_{analysis_params_general.slice_type}'
		figure_name = f'bundle_no_{bundle_no}_s{analysis_params_general.slice_type}.tif'

		my_help.check_dir(os.path.join(paths.fig_out_folder_path))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1, subfolder2))
		plt.savefig(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1, subfolder2, figure_name), bbox_inches='tight')
	else:
		plt.show()
	
	return fig


### plot heat-map of standardized density maps of GFP and RFP channel against the raw image.
def plot_bundle_vs_matrix(bundle_no, bundles_df, image, intensity_matrix, fig_params, thr_params, **kwargs):
	"""
	Function: plot heat-map of standardized density maps of GFP and RFP channel against the raw image.
	Input: 
	- bundle_no: int. Bundle No. for bundle-of-interest.
	- bundles_df: DataFrame. Contains heels and targets info.
	- image: numpy array. intensity-scaled raw image matrix.
	- intensity_matrix: numpy array. standardized density map matries for each channel and for each bundle.
	- fig_params: list. Contains: pp_i (parameters to passed on for intensity calculation function for bundle-of-interest), img_name (name of the original raw image), fig_name (name of the output figure)
	- thr_params: list. Values of the thresholds to mask heat-map, name of the thresholds, number of plot rows.
	- kwargs: additional settings.
		- is_label_off: Boolean. Whether to turn-off labels for heat-map.
		- is_save: Boolean. Whether to save output figure. 
		- is_true_x_tick: Boolean. Whether to use relative length value as x-axis tick.
	Output: figure
	"""

	### parameters from settings
	paths = settings.paths
	analysis_params_general = settings.analysis_params_general
	matching_info = settings.matching_info

	### unravel parameters
	params, folder_name, figure_name = fig_params
	thrs, thr_names, num_norm_channels = thr_params
	

	### setting seaborn parameters
	sns.set(font_scale=1.4)
	sns.set_style("ticks")
	
	z_offset, num_x_section, r_z, phis, center_point, ori_y_ticks, radius = params
	
	phi_start = phis[0]
	phi_end = phis[-1]
					   
	### X and Y tick settings
	if(kwargs['is_true_x_tick']):
		ori_x_ticks = np.round(np.linspace(0, analysis_params_general.radius_expanse_ratio, intensity_matrix.shape[2]), 2)
		x_ticks = ori_x_ticks
	else:
		x_ticks = np.arange(intensity_matrix.shape[1])

	x_ticks = get_tick_list(1, x_ticks, len(x_ticks)//3)
	y_ticks = get_tick_list(1, ori_y_ticks, len(ori_y_ticks)//5)

	# print(x_ticks, y_ticks)
	

	### R heels info
	r_heel_coords = my_help.get_heel_coords(bundle_no, bundles_df)

	### targets info
	ind_targets, coord_Tcs = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'c')
	_, coord_Te1s = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'e1')
	_, coord_Te2s = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'e2')
	_, target_ellipses = my_help.get_targets_info(bundle_no, bundles_df, return_type = 'ellipse')
	coords = np.concatenate((coord_Tcs, coord_Te1s, coord_Te2s))

	### image boundary
	image_bd = (image.shape[2], image.shape[1])
	xlim = [max(np.floor(coords.min(axis = 0))[0] - 30, 0), min(np.ceil(coords.max(axis = 0))[0] + 30, image_bd[0])]
	ylim = [max(np.floor(coords.min(axis = 0))[1] - 30, 0), min(np.ceil(coords.max(axis = 0))[1] + 30, image_bd[1])]

	image_xy = image[r_z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), 0].shape

	### plot
	plt.grid()   
	fig, axes = plt.subplots(num_norm_channels,3)
	fig.set_size_inches(8*5,9*5)
	fig.suptitle('Bundle No. ' + str(bundle_no), fontsize=80, fontweight='bold')

	i_fig = 0
	for ax in axes.flat:
		i_order = i_fig//3
		if(i_order%2 == 0):
			channel_no = 0
		else:
			channel_no = 1
		colormap = matching_info.channel_cmap[channel_no]

		### plot raw figure
		if(i_fig%3 == 0): 
			### get image crops for plot
			img_crop = image[r_z-z_offset : r_z+z_offset+1, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), channel_no]
			bundle_img = exposure.rescale_intensity(np.max(img_crop, axis = 0), in_range = 'image', out_range='dtype')
			
			## plot original bundle
			for i in range(6):
				ax.plot(r_heel_coords[i,0]-xlim[0], r_heel_coords[i,1]-ylim[0], color = matching_info.color_code[i+1], marker = 'o', markersize=12)

			## plot targets
			for i in matching_info.index_to_target_id.keys():
				ax.plot(coord_Tcs[i,0]-xlim[0], coord_Tcs[i,1]-ylim[0],'o', mec = matching_info.color_code[matching_info.index_to_target_id[i]], markersize=8, mew = 2.0, mfc = 'none')
				ax.add_patch(Ellipse((coord_Tcs[i,0]-xlim[0] , coord_Tcs[i,1]-ylim[0]),
									 width=target_ellipses[i,0], 
									 height=target_ellipses[i,1], 
									 angle = target_ellipses[i,2],
									 edgecolor=matching_info.color_code[matching_info.index_to_target_id[i]],
									 facecolor='none',
									 linewidth=2))

			## plot lines: boarder
			start_point = [ center_point[0] + radius * np.cos(phi_start) , center_point[1] + radius  * np.sin(phi_start) ]
			end_point = [ center_point[0] + radius * np.cos(phi_end) , center_point[1] + radius  * np.sin(phi_end) ]
			# Center-start
			ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')
			# Center-end
			ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')           

			## plot lines: -1 and 1
			start_point = coord_Tcs[2,:]
			end_point = coord_Tcs[5,:]
			ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')
			ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')            

			## plot circle
			circle = draw_circle(center_point, radius)
			ax.plot(circle[:,0] - xlim[0], circle[:,1] - ylim[0], '-', alpha = 0.5, color = 'w')

			# plot image
			ax.imshow(bundle_img, cmap=plt.cm.gray)

			# labels
			ax.set_title(matching_info.channel_mapping[channel_no] + ' channel', fontsize=16)

			if(kwargs['is_label_off']):
				ax.tick_params(axis='both', which='both', labelbottom='off', labelleft = 'off')
			else:
				ax.tick_params(axis = 'both', labelsize = 12)

			# standardize bundle orientation
			if(bundles_df.loc[bundle_no, 'Orientation_DV'] == 'L'):
				ax.invert_xaxis()
			if(bundles_df.loc[bundle_no, 'Orientation_AP'] == 'A'):
				ax.invert_yaxis()
		
		### plot mean intensity	
		elif(i_fig%3 == 1):         
				
			### matrix and thresholding mask
			matrix1 = np.max(intensity_matrix[channel_no,:,:,:], axis = 2)
			
			vmax = np.percentile(matrix1.flatten()[matrix1.flatten()>=0], 95)

			### plot
			if(thrs[i_order] > 0):
				mask1 = np.zeros_like(matrix1)
				mask1[matrix1<=thrs[i_order]] = True
			else:
				mask1 = np.zeros_like(matrix1)
				mask1[matrix1==0] = True
			g = sns.heatmap(matrix1, cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks, ax = ax, mask = mask1, vmin = 0, vmax = vmax)
			g.set_facecolor('lightgrey')

			ax.set_ylabel('Angle', fontsize=20)
			ax.set_xlabel('length from R-heel', fontsize=20)
			ax.set_title(f'Max, thr={thr_names[i_order]}', fontsize=20)
			ax.invert_yaxis()
		
		### plot max intensity	
		elif(i_fig%3 == 2): 
			
			### matrix
			matrix2 = np.mean(intensity_matrix[channel_no,:,:,:], axis = 2)
			vmax = np.percentile(matrix2.flatten()[matrix1.flatten()>=0], 95)
			
			### mask and plot
			if(thrs[i_order] > 0):
				mask1 = np.zeros_like(matrix1)
				mask1[matrix2<=thrs[i_order]] = True
			else:
				mask1 = np.zeros_like(matrix1)
				mask1[matrix2==0] = True
			g = sns.heatmap(matrix2, cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks, ax = ax, mask = mask1, vmin = 0, vmax = vmax)  
			g.set_facecolor('lightgrey')

			ax.set_ylabel('Angle', fontsize=20)
			ax.set_xlabel('length from R-heel', fontsize=20)
			ax.set_title(f'Mean, thr={thr_names[i_order]}', fontsize=20)
			ax.invert_yaxis()
		
		i_fig += 1
		
	plt.subplots_adjust(wspace = 0.2, hspace=0.3)
	
	if(kwargs['is_save']):
		plt.ioff()
		subfolder1 = 'HeatMap'
		subfolder2 = 'slice_type_' + str(analysis_params_general.slice_type)
		
		my_help.check_dir(os.path.join(paths.fig_out_folder_path))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1, subfolder2))
		
		plt.savefig(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1, subfolder2, figure_name), dpi=72, bbox_inches='tight')
	
	return fig


### plot polar density map of standardized density maps of GFP and RFP channels against the raw image.
def plot_polar(bundle_no, bundles_df, image, channel_no, matrix, fig_params, rel_points, **kwargs):

	"""
	Function: plot heat-map of standardized density maps of GFP and RFP channel against the raw image.
	Input: 
	- bundle_no: int. Bundle No. for bundle-of-interest.
	- bundles_df: DataFrame. Contains heels and targets info.
	- image: numpy array. intensity-scaled raw image matrix.
	- channeo_no: int. plotting RFP (0) or GFP (0) channel.
	- matrix: numpy array. standardized density map matries for each channel for bundle-of-interest.
	- rel_points: dictionary. relative length for target positions and heel positions
	- fig_params: list. Contains: pp_i (parameters to passed on for intensity calculation function for bundle-of-interest), img_name (name of the original raw image)
	- kwargs: additional parameters.
		- is_label_off: Boolean. Whether to turn-off labels for heat-map.
		- is_save: Boolean. Whether to save output figure. 
	Output: figure
	"""

	### decompose parameters
	if('is_label_off' in kwargs.keys()):
		is_label_off = kwargs['is_label_off']
	else:
		is_label_off = False
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False

	params, img_name = fig_params

	z_offset, num_x_section, r_z, phis, center_point, ori_y_ticks, radius = params
	
	paths = settings.paths
	matching_info = settings.matching_info
	analysis_params_general = settings.analysis_params_general

	phi_start = phis[0]
	phi_end = phis[-1]

	colormap = plt.get_cmap(matching_info.channel_cmap[channel_no])
	colormap.set_bad('lightgrey')

	analysis_params = analysis_params_general.num_angle_section, analysis_params_general.num_outside_angle, analysis_params_general.num_x_section, analysis_params_general.z_offset, analysis_params_general.radius_expanse_ratio

	### values for subfigures
	#### subfigure 1: raw image.
	r_heel_coords, coord_Tcs, target_ellipses, bundle_img, xlim, ylim = get_imshow_values(bundles_df, bundle_no, image, r_z, analysis_params_general.z_offset, channel_no)
	
	#### subfigure2: max projection of z of density map.
	mm = np.max(matrix[channel_no, :, :, :], axis = 2)
	thetav, rv, z1, norm1, target_grid_polar = get_polar_plot_values(analysis_params, channel_no, mm, colormap, rel_points)
	
	#### subfigure3: mean over z of density map.
	mm = np.mean(matrix[channel_no, :, :, :], axis = 2)
	thetav, rv, z2, norm2, target_grid_polar = get_polar_plot_values(analysis_params, channel_no, mm, colormap, rel_points)

	### plotting
	sns.set_style('ticks')
	fig = plt.figure(figsize = (18,6))

	#### subfigure 1: plot raw image
	ax = fig.add_subplot(131, polar = False)

	##### plot original bundle
	for i in range(6):
		ax.plot(r_heel_coords[i,0]-xlim[0], r_heel_coords[i,1]-ylim[0], color = matching_info.color_code[i+1], marker = 'o', markersize=12)

	##### plot targets
	for i in matching_info.index_to_target_id.keys():
		ax.plot(coord_Tcs[i,0]-xlim[0], coord_Tcs[i,1]-ylim[0],'o', mec = matching_info.color_code[matching_info.index_to_target_id[i]], markersize=8, mew = 2.0, mfc = 'none')
		ax.add_patch(Ellipse((coord_Tcs[i,0]-xlim[0] , coord_Tcs[i,1]-ylim[0]),
							 width=target_ellipses[i,0], 
							 height=target_ellipses[i,1], 
							 angle = target_ellipses[i,2],
							 edgecolor=matching_info.color_code[matching_info.index_to_target_id[i]],
							 facecolor='none',
							 linewidth=2))

	##### plot lines: boarder
	start_point = [ center_point[0] + radius * np.cos(phi_start) , center_point[1] + radius * np.sin(phi_start) ]
	end_point = [ center_point[0] + radius * np.cos(phi_end) , center_point[1] + radius * np.sin(phi_end) ]
	###### Center-start
	ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')
	###### Center-end
	ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')            
	##### plot lines: -1 and 1
	start_point = coord_Tcs[2,:]
	end_point = coord_Tcs[5,:]
	ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')
	ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')            

	##### plot circle
	circle = draw_circle(center_point, radius)
	ax.plot(circle[:,0] - xlim[0], circle[:,1] - ylim[0], '--', alpha = 0.5, color = 'w')

	##### plot image
	ax.imshow(bundle_img, cmap=plt.cm.gray)

	##### labels
	ax.set_title(matching_info.channel_mapping[channel_no] + ' channel', fontsize=16)

	if(is_label_off):
		ax.tick_params(axis='both', which='both', labelbottom='off', labelleft = 'off')
	else:
		ax.tick_params(axis = 'both', labelsize = 12)

	##### standardize bundle orientation
	if(bundles_df.loc[bundle_no, 'Orientation_DV'] == 'L'):
		ax.invert_xaxis()
	if(bundles_df.loc[bundle_no, 'Orientation_AP'] == 'A'):
		ax.invert_yaxis()


	#### subfigure 2:  polar plot: max
	ax2 = fig.add_subplot(132, polar = True)

	##### plot value
	mask = z1 == 0
	zm1 = ma.masked_array(z1, mask=mask)
	vmax = np.percentile(zm1, 99)    

	sc = ax2.pcolormesh(thetav, rv, zm1, cmap=colormap, vmin = 0, vmax = vmax)
	
	##### plot target position
	for i in [2, 3, 5]:
		ax2.plot(target_grid_polar['c'][i,0], target_grid_polar['c'][i,1], 'o', color = matching_info.color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 3, mfc = 'none')
		ax2.plot([target_grid_polar['l'][i,0],target_grid_polar['h'][i,0]],
				[target_grid_polar['l'][i,1],target_grid_polar['h'][i,1]],
				'-',
				color = matching_info.color_code[matching_info.index_to_target_id[i]],
				linewidth = 2)

	##### plot angle reference
	ax2.plot([target_grid_polar['c'][6,0], target_grid_polar['c'][matching_info.target_id_to_index[3],0]], [target_grid_polar['c'][6,1], target_grid_polar['c'][matching_info.target_id_to_index[3],1]], '--', color = '0.5')
	ax2.plot([target_grid_polar['c'][6,0], target_grid_polar['c'][matching_info.target_id_to_index[7],0]], [target_grid_polar['c'][6,1], target_grid_polar['c'][matching_info.target_id_to_index[7],1]], '--', color = '0.5')

	##### set polar to pie
	ax2.set_thetamin(-50)
	ax2.set_thetamax(50)
	ax2.set_title('Max Projection')

	##### color bar for polar plot
	cNorm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)   #-- Defining a normalised scale
	ax5 = fig.add_axes([0.6, 0.2, 0.02, 0.6])       #-- Creating a new axes at the right side
	cb1 = matplotlib.colorbar.ColorbarBase(ax5, norm=cNorm, cmap=colormap)    #-- Plotting the colormap in the created axes
	fig.subplots_adjust(left=0.0,right=0.95)

	#### subfigure 3: polar plot: mean
	ax3 = fig.add_subplot(133, polar = True)

	##### plot value
	mask = z2 ==0
	zm2 = ma.masked_array(z2, mask=mask)
	vmax = np.percentile(zm2, 99)
	sc = ax3.pcolormesh(thetav, rv, zm2, cmap=colormap, vmin = 0, vmax = vmax)

	##### plot target position
	for i in [2, 3, 5]:
		ax3.plot(target_grid_polar['c'][i,0], target_grid_polar['c'][i,1], 'o', color = matching_info.color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 3, mfc = 'none')
		ax3.plot([target_grid_polar['l'][i,0],target_grid_polar['h'][i,0]],
				[target_grid_polar['l'][i,1],target_grid_polar['h'][i,1]],
				'-',
				color = matching_info.color_code[matching_info.index_to_target_id[i]],
				linewidth = 2)

	##### plot angle reference
	ax3.plot([target_grid_polar['c'][6,0], target_grid_polar['c'][matching_info.target_id_to_index[3],0]], [target_grid_polar['c'][6,1], target_grid_polar['c'][matching_info.target_id_to_index[3],1]], '--', color = '0.5')
	ax3.plot([target_grid_polar['c'][6,0], target_grid_polar['c'][matching_info.target_id_to_index[7],0]], [target_grid_polar['c'][6,1], target_grid_polar['c'][matching_info.target_id_to_index[7],1]], '--', color = '0.5')

	##### set polar to pie
	ax3.set_thetamin(-50)
	ax3.set_thetamax(50)
	ax3.set_title('Mean Projection')

	##### color bar for polar plot
	cNorm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)   #-- Defining a normalised scale
	ax4 = fig.add_axes([0.9, 0.2, 0.02, 0.6])       #-- Creating a new axes at the right side
	cb1 = matplotlib.colorbar.ColorbarBase(ax4, norm=cNorm, cmap=colormap)    #-- Plotting the colormap in the created axes
	fig.subplots_adjust(left=0.0,right=0.95)
	
	### saving figure
	if(is_save):
		plt.ioff()
		folder_name = img_name
		subfolder1 = 'PolarPlot'
		subfolder2 = f'slice_type_{analysis_params_general.slice_type}'
		figure_name = f'bundle_no_{bundle_no}_channel_{matching_info.channel_mapping[channel_no]}.tif'

		my_help.check_dir(os.path.join(paths.fig_out_folder_path))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1))
		my_help.check_dir(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1, subfolder2))
		plt.savefig(os.path.join(paths.fig_out_folder_path, folder_name, subfolder1, subfolder2, figure_name), dpi=72, bbox_inches='tight')
	
	return fig