# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   sf942274
# @Last Modified time: 2019-10-18 00:17:26


import io, os, sys, types

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import numpy as np
from numpy.linalg import eig, inv

import math

from scipy import interpolate, spatial, stats

import seaborn as sns

import skimage.io as skiIo
from skimage import exposure, img_as_float, filters

from sklearn import linear_model
from sklearn import metrics


import data_quantification_function_helper as my_help
import data_quantification_function_intensity_calculation as my_int
import data_quantification_settings as settings



def get_all_rcell_coords(bundle_no, bundles_df):
	allRcellCoords = np.zeros((6,2))
	allRcellCoords[:,0] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_X_R', True)])
	allRcellCoords[:,1] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_Y_R', True)])
	return allRcellCoords

def draw_circle(center, radius):
	theta = np.linspace(0, 2*np.pi, 100)

	# compute x1 and x2
	x = radius*np.cos(theta) + center[0]
	y = radius*np.sin(theta) + center[1]
	circle = np.column_stack((x,y))
	return circle

### reduce x and y axis labels.
### type 1: specify number of labels remained.
### type 2: specify which label should be kept.
def get_tick_list(tickType, ori_tick, arg2):
	if(tickType == 1):
		numOfLabels = arg2
		k = np.ceil((len(ori_tick) - 1)/(numOfLabels)).astype(int)
		nk = np.floor(len(ori_tick)/k).astype(int)

		trueLabelList = []
		i = 0
		ti = 0 + i*k
		while ti < len(ori_tick):
			trueLabelList.append(0 + i*k)
			i += 1
			ti = 0 + i*k
			
	elif(tickType == 2):
		must_lists = arg2
		trueLabelList = np.where(np.isin(ori_tick,must_lists))[0]
		
	TickList = []
	for i in range(len(ori_tick)):
		if i in trueLabelList:
			TickList.append(ori_tick[i].astype(str))
		else:
			TickList.append('')            
	
	return TickList

def get_slice_params_for_polar_plot(analysis_params, slicingParams):
	num_angleSection, num_outsideAngle, num_x_section, z_offset, radius_expanse_ratio = analysis_params
	SliceZeroPoint, SliceOnePoint, CutOffPoint, center_point = slicingParams
	
	radius = np.linalg.norm( center_point - CutOffPoint )
	angStart2R = np.arctan2( SliceZeroPoint[1] - center_point[1], SliceZeroPoint[0] - center_point[0] )
	angEnd2R = np.arctan2( SliceOnePoint[1] - center_point[1], SliceOnePoint[0] - center_point[0])
	
	phiRange = my_int.inner_angle(SliceOnePoint - center_point, SliceZeroPoint - center_point, True)
	phi_unit = phiRange/num_angleSection
	
	if(((-np.pi <= angStart2R <= -0.5*np.pi) | (-np.pi <= angEnd2R <= -0.5*np.pi)) & (angStart2R*angEnd2R < 1) ):
		if((-np.pi <= angStart2R <= -0.5*np.pi) & (-np.pi <= angEnd2R <= -0.5*np.pi)):
			phi_start = min(angStart2R, angEnd2R) - num_outsideAngle * phi_unit
			phi_end = max(angStart2R, angEnd2R) + num_outsideAngle * phi_unit
		else:
			phi_start = max(angStart2R, angEnd2R) - num_outsideAngle * phi_unit
			phi_end = min(angStart2R, angEnd2R) + num_outsideAngle * phi_unit
	else:
		phi_start = min(angStart2R, angEnd2R) - num_outsideAngle * phi_unit
		phi_end = max(angStart2R, angEnd2R) + num_outsideAngle * phi_unit

	phi_start = my_int.angle_normalization(phi_start)
	phi_end = my_int.angle_normalization(phi_end)

	phis = my_int.getPhis(phi_start, phi_end, num_angleSection + num_outsideAngle*2 + 2)

	if(my_int.smallest_angle(angStart2R, phis[-1]) < my_int.smallest_angle(angStart2R, phis[0])):
		phis = np.flip(phis, axis = 0)
	
	rs = np.linspace(0, radius_expanse_ratio, num_x_section + 2)
	
	return rs, phis

def get_target_grid():
	## Distances
	dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
	dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
	dT2T5 = dT3T3_ = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
	dT0T3 = dT0T3_ = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5

	## Angles (in radius)
	aT2T0T5 = math.radians(80)
	aT0T5T4 = math.radians(100)
	aT3T0T3_ = math.acos((dT0T3 ** 2 + dT0T3_ ** 2 - dT3T3_ ** 2)/(2*dT0T3*dT0T3_))

	## normalized axis
	dT0T4n = 1
	dT2T5n = dT2T5*dT0T4n/dT0T4

	### normalized target grid positions
	T0 = np.array((0,0))
	T2 = np.array((dT0T4n/2, dT2T5n/2))
	T5 = np.array((dT0T4n/2, -dT2T5n/2))
	T4 = np.array((dT0T4n,0))
	T3 = np.array((dT0T4n + dT0T4n/2, dT2T5n/2))
	T3_ = np.array((dT0T4n + dT0T4n/2, -dT2T5n/2))
	targetGridPos = np.stack((T0, T2, T3, T4, T5, T3_), axis = 0)
	
	return targetGridPos

def get_target_grid_polar():
	dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
	dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
	dT2T5 = dT3T3_ = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
	dT0T3 = dT0T3_ = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5
	
	aT0T2 = math.radians(80)/2
	aT0T5 = - math.radians(80)/2
	aT0T3 = math.acos((dT0T3 ** 2 + dT0T3_ ** 2 - dT3T3_ ** 2)/(2*dT0T3*dT0T3_))/2
	aT0T3_ = - aT0T3
	aT0T4 = 0
	
	## normalized axis
	dT0T4n = 1
	dT0T2n = dT0T2*dT0T4n/dT0T4
	dT0T3n = dT0T3*dT0T4n/dT0T4
	
	T0 = np.array((0,0))
	T2 = np.array((aT0T2, dT0T2n))
	T3 = np.array((aT0T3, dT0T3n))
	T4 = np.array((aT0T4, dT0T4n))
	T5 = np.array((aT0T5, dT0T2n))
	T3_ = np.array((aT0T3_, dT0T3n))
	
	target_grid_polar = np.stack((T0, T2, T3, T4, T5, T3_), axis = 0)
	
	return target_grid_polar


def get_target_grid_polar_new(rel_points):
	matching_info = settings.matching_info
	dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
	dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
	dT2T5 = dT3T3_ = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
	dT0T3 = dT0T3_ = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5
	
	aT0T2 = math.radians(80)/2
	aT0T5 = - math.radians(80)/2
	aT0T3 = math.acos((dT0T3 ** 2 + dT0T3_ ** 2 - dT3T3_ ** 2)/(2*dT0T3*dT0T3_))/2
	aT0T3_ = - aT0T3
	aT0T4 = 0
	
	## normalized axis
	dT0T4n = 1
	dT0T2n = dT0T2*dT0T4n/dT0T4
	dT0T3n = dT0T3*dT0T4n/dT0T4
	
	T0 = np.array((0,-rel_points[matching_info.target_id_to_index[0]]))
	T2 = np.array((aT0T2, rel_points[matching_info.target_id_to_index[2]]))
	T3 = np.array((aT0T3, rel_points[matching_info.target_id_to_index[3]]))
	T4 = np.array((aT0T4, rel_points[matching_info.target_id_to_index[4]]))
	T5 = np.array((aT0T5, rel_points[matching_info.target_id_to_index[5]]))
	T3_ = np.array((aT0T3_, rel_points[matching_info.target_id_to_index[7]]))
	C0 = np.array((0, rel_points[6]))
	
	target_grid_polar = np.stack((T0, T2, T3, T4, T5, T3_, C0), axis = 0)
	
	return target_grid_polar


def get_polar_plot_values(analysis_params, channel_no, matrix, cmap, radius_expanse_ratio, rel_points):
	matching_info = settings.matching_info
	target_grid_polar = get_target_grid_polar_new(rel_points)
	targetGridPos = get_target_grid()
	CutOffPoint = 1 * radius_expanse_ratio
	bundleParams = targetGridPos[matching_info.target_id_to_index[7]], targetGridPos[matching_info.target_id_to_index[3]], CutOffPoint, targetGridPos[matching_info.target_id_to_index[0]] 

	rs, phis = get_slice_params_for_polar_plot(analysis_params, bundleParams)
	thetav, rv = np.meshgrid(phis, rs)
	z = matrix.transpose()

	levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	return thetav, rv, z, norm, target_grid_polar

def get_imshow_values(bundles_df, bundle_no, image, r_z, z_offset, channel_no):
	matching_info = settings.matching_info
	### R heels info
	r_heel_coords = np.zeros((6,2))
	r_heel_coords[:,0] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_X_R', True)])
	r_heel_coords[:,1] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_Y_R', True)])

	### targets info
	ind_targets, coord_targets = my_help.getTargetCoords(bundle_no, bundles_df, matching_info.index_to_target_id)
	coord_rs = my_help.getRxCoords(bundle_no, bundles_df, ind_targets, 4)       
	coords = np.concatenate((coord_targets, coord_rs))

	### image boundary
	image_bd = (image.shape[2], image.shape[1])
	xlim = [max(np.floor(coords.min(axis = 0))[0] - 21, 0), min(np.ceil(coords.max(axis = 0))[0] + 21, image_bd[0])]
	ylim = [max(np.floor(coords.min(axis = 0))[1] - 21, 0), min(np.ceil(coords.max(axis = 0))[1] + 21, image_bd[1])]

	img_crop = image[r_z-z_offset : r_z+z_offset+1, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), channel_no]
	bundle_img = exposure.rescale_intensity(np.max(img_crop, axis = 0), in_range = 'image', out_range='dtype')

	# bundle_img = exposure.rescale_intensity(image[r_z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), channel_no], in_range = 'image', out_range='dtype')

	return r_heel_coords, coord_targets, bundle_img, xlim, ylim

""" ========================= Plotting =========================== """
def plotIndividualBundles(bundle_no, bundles_df, image_norm, xRatio, yRatio, z_offset, plotSettings, matching_info):
	
	### Unravel plot settings
	isPlotR3Line, isPlotR4Line, isPlotR4s, is_label_off = plotSettings
	matching_info.index_to_target_id, matching_info.color_code, matching_info.channel_mapping, matching_info.channel_cmap, matching_info.target_id_to_index = matching_info
	
	### R heels info
	R4_Z = int(bundles_df.loc[bundle_no,'coord_Z_R4']) - 1
	print(R4_Z)
	r_heel_coords = np.zeros((6,2))
	r_heel_coords[:,0] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_X_R', True)])
	r_heel_coords[:,1] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_Y_R', True)])

	### targets info
	ind_targets, coord_targets = my_help.getTargetCoords(bundle_no, bundles_df, matching_info.index_to_target_id)
	coordR4s = my_help.getRxCoords(bundle_no, bundles_df, ind_targets, 4)
	coords = np.concatenate((coord_targets, coordR4s))

	### image boundary
	image_bd = (image_norm.shape[2], image_norm.shape[1])
	xlim = [max(np.floor(coords.min(axis = 0))[0] - 50, 0), min(np.ceil(coords.max(axis = 0))[0] + 50, image_bd[0])]
	ylim = [max(np.floor(coords.min(axis = 0))[1] - 50, 0), min(np.ceil(coords.max(axis = 0))[1] + 50, image_bd[1])]
	# print(xlim, ylim)

	image_xy = image_norm[R4_Z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), 0].shape

	### get image crops for plot
	bundle_img = np.empty(image_xy + (image_norm.shape[-1],), dtype=image_norm.dtype, order='C')
	for i in range(image_norm.shape[-1]):
		img_crop = image_norm[R4_Z-z_offset : R4_Z+z_offset+1, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), i]
		bundle_img[:,:,i] = exposure.rescale_intensity(np.max(img_crop, axis = 0), in_range = 'image', out_range='dtype')

	### plot
	fig, axes = plt.subplots(2, 4, figsize = (18,8))
	fig.suptitle('Bundle No. ' + str(bundle_no), fontsize=18, fontweight='bold')

	for plt_i in range(image_norm.shape[-1]):
		ax = axes.ravel()[plt_i]
		
		# plot original bundle
		for i in range(6):
			ax.plot(r_heel_coords[i,0]-xlim[0], r_heel_coords[i,1]-ylim[0], color = matching_info.color_code[i+1], marker = 'o', markersize=12)

		# plot targets
		for i in matching_info.index_to_target_id.keys():
			ax.plot(coord_targets[i,0]-xlim[0], coord_targets[i,1]-ylim[0],'o', mec = matching_info.color_code[matching_info.index_to_target_id[i]], markersize=8, mew = 2.0, mfc = 'none')

		# plot R4s
		if(isPlotR4s):
			for i in range(1,len(coordR4s)):
				ax.plot(coordR4s[i,0]-xlim[0], coordR4s[i,1]-ylim[0], color = matching_info.color_code[4], marker = 'o', markersize=7)

		# plot lines
		if(isPlotR3Line):
			# lines associated with R3 & R4
			# R3-T3
			ax.plot([r_heel_coords[2,0]-xlim[0], coord_targets[2,0]-xlim[0]], [r_heel_coords[2,1]-ylim[0], coord_targets[2,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[3])
			# R3-T7
			ax.plot([r_heel_coords[2,0]-xlim[0], coord_targets[5,0]-xlim[0]], [r_heel_coords[2,1]-ylim[0], coord_targets[5,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[3])
		if(isPlotR4Line):
			# R4-T3
			ax.plot([r_heel_coords[3,0]-xlim[0], coord_targets[2,0]-xlim[0]], [r_heel_coords[3,1]-ylim[0], coord_targets[2,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[4])
			# R4-T7
			ax.plot([r_heel_coords[3,0]-xlim[0], coord_targets[5,0]-xlim[0]], [r_heel_coords[3,1]-ylim[0], coord_targets[5,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[4])            

		# plot image
		ax.imshow(bundle_img[:,:,plt_i], cmap=plt.cm.gray)

		# plot axis
		ax.set_title(matching_info.channel_mapping[plt_i], fontsize=16)

	#         plt_i += 1

		if(is_label_off):
			ax.tick_params(axis='both', which='both', labelbottom='off', labelleft = 'off')
		else:
			ax.tick_params(axis = 'both', labelsize = 12)


def plot_bundle_vs_matrix_all(bundle_no, bundles_df, image, intensity_matrix, fig_params, tick_params, plot_options, **kwarg):
	## unravel parameters
	analysis_params_general.center_type, params, figure_prefix, folder_name, figure_name, slice_type, radius_expanse_ratio = fig_params
	thrs, thr_function, num_norm_channels = plot_options
	tick_type_x, tick_type_y, tick_arg2_x, tick_arg2_y = tick_params
	matching_info = settings.matching_info
	# matching_info.index_to_target_id, matching_info.color_code, matching_info.channel_mapping, matching_info.channel_cmap, matching_info.target_id_to_index = matching_info

	## setting seaborn parameters
	sns.set(font_scale=1.4)
	sns.set_style("ticks")
	
	z_offset, num_x_section, r_z, phis, center_point, ori_y_ticks, radius = params
	# print(center_point)
	
	phi_start = phis[0]
	phi_end = phis[-1]
					   
	### X and Y tick settings
	if(kwarg['is_ori_tick']):
		if(kwarg['is_true_x_tick']):
			ori_x_ticks = np.round(np.linspace(0, radius_expanse_ratio, intensity_matrix.shape[2]), 2)
			# print(ori_x_ticks)
			x_ticks = ori_x_ticks
		else:
			x_ticks = np.arange(intensity_matrix.shape[1])
			# print(x_ticks)
		y_ticks = ori_y_ticks
	else:
		if(kwarg['is_true_x_tick']):
			ori_x_ticks = np.round(np.linspace(0, radius_expanse_ratio, intensity_matrix.shape[2]), 2)
			# print(ori_x_ticks)
			x_ticks = get_tick_list(tick_type_x, ori_x_ticks, tick_arg2_x)
		else:
			x_ticks = np.arange(intensity_matrix.shape[1])
			# print(x_ticks)
		
		y_ticks = get_tick_list(tick_type_y, ori_y_ticks, tick_arg2_y)
	# print(ori_y_ticks)
	# print(y_ticks)
	# print(x_ticks)
	

	### R heels info
	r_heel_coords = np.zeros((6,2))
	r_heel_coords[:,0] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_X_R', True)])
	r_heel_coords[:,1] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_Y_R', True)])

	### targets info
	ind_targets, coord_targets = my_help.getTargetCoords(bundle_no, bundles_df, matching_info.index_to_target_id)
	coord_rs = my_help.getRxCoords(bundle_no, bundles_df, ind_targets, 4)        
	coords = np.concatenate((coord_targets, coord_rs))

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
		channel_no = i_fig//3
		colormap = matching_info.channel_cmap[channel_no]

		if(i_fig%3 == 0): ### plot raw figure
			### get image crops for plot
			img_crop = image[r_z-z_offset : r_z+z_offset+1, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), channel_no]
			bundle_img = exposure.rescale_intensity(np.max(img_crop, axis = 0), in_range = 'image', out_range='dtype')
			
			## plot original bundle
			for i in range(6):
				ax.plot(r_heel_coords[i,0]-xlim[0], r_heel_coords[i,1]-ylim[0], color = matching_info.color_code[i+1], marker = 'o', markersize=12)

			## plot targets
			for i in matching_info.index_to_target_id.keys():
				ax.plot(coord_targets[i,0]-xlim[0], coord_targets[i,1]-ylim[0],'o', mec = matching_info.color_code[matching_info.index_to_target_id[i]], markersize=8, mew = 2.0, mfc = 'none')

			## plot lines: boarder
			start_point = [ center_point[0] + radius * np.cos(phi_start) , center_point[1] + radius  * np.sin(phi_start) ]
			end_point = [ center_point[0] + radius * np.cos(phi_end) , center_point[1] + radius  * np.sin(phi_end) ]
			# Center-start
			ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')
			# Center-end
			ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')            

			## plot lines: -1 and 1
			start_point = coord_targets[2,:]
			end_point = coord_targets[5,:]
			ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')
			ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')            

			## plot circle
			circle = draw_circle(center_point, radius)
			ax.plot(circle[:,0] - xlim[0], circle[:,1] - ylim[0], '-', alpha = 0.5, color = 'w')

			# plot image
			ax.imshow(bundle_img, cmap=plt.cm.gray)

			# labels
			ax.set_title(matching_info.channel_mapping[channel_no] + ' channel', fontsize=16)

			if(kwarg['is_label_off']):
				ax.tick_params(axis='both', which='both', labelbottom='off', labelleft = 'off')
			else:
				ax.tick_params(axis = 'both', labelsize = 12)

			# standardize bundle orientation
			if(bundles_df.loc[bundle_no, 'Orientation_DV'] == 'L'):
				ax.invert_xaxis()
			if(bundles_df.loc[bundle_no, 'Orientation_AP'] == 'A'):
				ax.invert_yaxis()
				
		elif(i_fig%3 == 1): # plot mean intensity            
				
			### matrix and thresholding mask
			matrix1 = np.max(intensity_matrix[channel_no,:,:,:], axis = 2)
			# print(matrix1)
			# print(matrix1.shape)
			# print(y_ticks, x_ticks)
			vmax = np.percentile(matrix1.flatten(), 95)
			
			### plot
			if(thrs[channel_no] > 0):
				mask1 = np.zeros_like(matrix1)
				mask1[matrix1<=thrs[channel_no]] = True
				with sns.axes_style("dark"):
					sns.heatmap(matrix1, cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks, ax = ax, mask = mask1, vmax = vmax)
			else:
				sns.heatmap(matrix1, cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks, ax = ax, vmax = vmax)
			ax.set_ylabel('Angle', fontsize=20)
			ax.set_xlabel('length from R-heel', fontsize=20)
			ax.set_title('Max projection over z-stack', fontsize=20)
			ax.invert_yaxis()
			
		elif(i_fig%3 == 2): # plot max intensity            
			
			### matrix
			matrix2 = np.mean(intensity_matrix[channel_no,:,:,:], axis = 2)
			vmax = np.percentile(matrix2.flatten(), 95)
			
			### mask and plot
			if(thrs[channel_no] > 0):
				mask1 = np.zeros_like(matrix1)
				mask1[matrix2<=thrs[channel_no]] = True
				with sns.axes_style("dark"):
					sns.heatmap(matrix2, cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks, ax = ax, mask = mask1, vmax = vmax)
			else:
				sns.heatmap(matrix2, cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks, ax = ax, vmax = vmax)            
			ax.set_ylabel('Angle', fontsize=20)
			ax.set_xlabel('length from R-heel', fontsize=20)
			ax.set_title('Mean of z-stack', fontsize=20)
			ax.invert_yaxis()
		
		i_fig += 1
		
	plt.subplots_adjust(wspace = 0.2, hspace=0.3)
	
	if(kwarg['is_save']):
		plt.ioff()
		subfolder1 = 'HeatMap'
		subfolder2 = 'slice_type_' + str(slice_type) + '_analysis_params_general.center_type' + str(analysis_params_general.center_type) + '_ThrType' + str(thr_function)
		# figure_name = 'bundle_no_' + str(bundle_no) + '_' + str(thr_function)
		
		my_help.check_dir(os.path.join(figure_prefix))
		my_help.check_dir(os.path.join(figure_prefix, folder_name))
		my_help.check_dir(os.path.join(figure_prefix, folder_name, subfolder1))
		my_help.check_dir(os.path.join(figure_prefix, folder_name, subfolder1, subfolder2))
		
		plt.savefig(os.path.join(figure_prefix, folder_name, subfolder1, subfolder2, figure_name), dpi=300, bbox_inches='tight')
	
	return fig

def plot_polar(bundle_no, bundles_df, image, channel_no, matrix, fig_params, matching_info, rel_points, **kwarg):

	##### decompose parameters
	params, figure_prefix, img_name, radius_expanse_ratio, analysis_params_general.center_type, slice_type = fig_params

	z_offset, num_x_section, r_z, phis, center_point, ori_y_ticks, radius = params
	
	# is_label_off, isSave = plot_options

	matching_info = settings.matching_info
	analysis_params_general = settings.analysis_params_general
	# matching_info.index_to_target_id, matching_info.color_code, matching_info.channel_mapping, matching_info.channel_cmap, matching_info.target_id_to_index = matching_info

	phi_start = phis[0]
	phi_end = phis[-1]

	colormap = plt.get_cmap(matching_info.channel_cmap[channel_no])

	analysis_params = analysis_params_general.num_angle_section, analysis_params_general.num_outside_angle, analysis_params_general.num_x_section, analysis_params_general.z_offset, analysis_params_general.radius_expanse_ratio[analysis_params_general.center_type]

	##### values for subfigures
	#### subfigure 1
	r_heel_coords, coord_targets, bundle_img, xlim, ylim = get_imshow_values(bundles_df, bundle_no, image, r_z, analysis_params_general.z_offset, channel_no)
	#### subfigure2: max projection of z
	mm = np.max(matrix[channel_no, :, :, :], axis = 2)
	thetav, rv, z1, norm1, target_grid_polar = get_polar_plot_values(analysis_params, channel_no, mm, colormap, matching_info.target_id_to_index, radius_expanse_ratio, rel_points)
	#### subfigure3: mean over z
	mm = np.mean(matrix[channel_no, :, :, :], axis = 2)
	thetav, rv, z2, norm2, target_grid_polar = get_polar_plot_values(analysis_params, channel_no, mm, colormap, matching_info.target_id_to_index, radius_expanse_ratio, rel_points)

	##### plotting
	fig = plt.figure(figsize = (18,6))

	#### subfigure 1: plot raw image
	ax = fig.add_subplot(131, polar = False)

	## plot original bundle
	for i in range(6):
		ax.plot(r_heel_coords[i,0]-xlim[0], r_heel_coords[i,1]-ylim[0], color = matching_info.color_code[i+1], marker = 'o', markersize=12)

	## plot targets
	for i in matching_info.index_to_target_id.keys():
		ax.plot(coord_targets[i,0]-xlim[0], coord_targets[i,1]-ylim[0],'o', mec = matching_info.color_code[matching_info.index_to_target_id[i]], markersize=8, mew = 2.0, mfc = 'none')

	## plot lines: boarder
	start_point = [ center_point[0] + radius * np.cos(phi_start) , center_point[1] + radius * np.sin(phi_start) ]
	end_point = [ center_point[0] + radius * np.cos(phi_end) , center_point[1] + radius * np.sin(phi_end) ]
	# Center-start
	ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')
	# Center-end
	ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '-', alpha = 0.5, color = 'w')            

	## plot lines: -1 and 1
	start_point = coord_targets[2,:]
	end_point = coord_targets[5,:]
	ax.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')
	ax.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')            

	
	## plot circle
	circle = draw_circle(center_point, radius)
	ax.plot(circle[:,0] - xlim[0], circle[:,1] - ylim[0], '--', alpha = 0.5, color = 'w')

	## plot image
	ax.imshow(bundle_img, cmap=plt.cm.gray)

	## labels
	ax.set_title(matching_info.channel_mapping[channel_no] + ' channel', fontsize=16)

	if(kwarg['is_label_off']):
		ax.tick_params(axis='both', which='both', labelbottom='off', labelleft = 'off')
	else:
		ax.tick_params(axis = 'both', labelsize = 12)

	## standardize bundle orientation
	if(bundles_df.loc[bundle_no, 'Orientation_DV'] == 'L'):
		ax.invert_xaxis()
	if(bundles_df.loc[bundle_no, 'Orientation_AP'] == 'A'):
		ax.invert_yaxis()

	#### parameters for subfigure 2 and 3

	#### subfigure 2:  polar plot: max
	ax2 = fig.add_subplot(132, polar = True)

	## plot value
	sc = ax2.pcolormesh(thetav, rv, z1, cmap=colormap, norm=norm1)

	## plot heel position
	if(analysis_params_general.center_type == 0):
		for i in [0,2,3,5]:
			ax2.plot(target_grid_polar[i,0], target_grid_polar[i,1], 'o', color = matching_info.color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 3, mfc = 'none')
	elif(analysis_params_general.center_type == 1):
		for i in [2,3,5]:
			ax2.plot(target_grid_polar[i,0], target_grid_polar[i,1], 'o', color = matching_info.color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 3, mfc = 'none')
	## plot angle reference
	if(analysis_params_general.center_type == 0):
		ax2.plot([target_grid_polar[0,0], target_grid_polar[matching_info.target_id_to_index[3],0]], [target_grid_polar[0,1], target_grid_polar[matching_info.target_id_to_index[3],1]], '--', color = '0.5')
		ax2.plot([target_grid_polar[0,0], target_grid_polar[matching_info.target_id_to_index[7],0]], [target_grid_polar[0,1], target_grid_polar[matching_info.target_id_to_index[7],1]], '--', color = '0.5')
	elif(analysis_params_general.center_type == 1):
		ax2.plot([target_grid_polar[6,0], target_grid_polar[matching_info.target_id_to_index[3],0]], [target_grid_polar[6,1], target_grid_polar[matching_info.target_id_to_index[3],1]], '--', color = '0.5')
		ax2.plot([target_grid_polar[6,0], target_grid_polar[matching_info.target_id_to_index[7],0]], [target_grid_polar[6,1], target_grid_polar[matching_info.target_id_to_index[7],1]], '--', color = '0.5')

	
	## set polar to pie
	ax2.set_thetamin(-50)
	ax2.set_thetamax(50)
	ax2.set_title('Max Projection')

	#### color bar for polar plot
	vmin,vmax = sc.get_clim()
	cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)   #-- Defining a normalised scale
	ax5 = fig.add_axes([0.6, 0.2, 0.02, 0.6])       #-- Creating a new axes at the right side
	cb1 = matplotlib.colorbar.ColorbarBase(ax5, norm=cNorm, cmap=colormap)    #-- Plotting the colormap in the created axes
	fig.subplots_adjust(left=0.0,right=0.95)

	#### subfigure 3: polar plot: mean
	ax3 = fig.add_subplot(133, polar = True)

	## plot value
	sc = ax3.pcolormesh(thetav, rv, z2, cmap=colormap, norm=norm2)

	## plot heel position
	if(analysis_params_general.center_type == 0):
		for i in [0,2,3,5]:
			ax3.plot(target_grid_polar[i,0], target_grid_polar[i,1], 'o', color = matching_info.color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 3, mfc = 'none')
	elif(analysis_params_general.center_type == 1):
		for i in [2,3,5]:
			ax3.plot(target_grid_polar[i,0], target_grid_polar[i,1], 'o', color = matching_info.color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 3, mfc = 'none')
	
	## plot angle reference
	if(analysis_params_general.center_type == 0):
		ax3.plot([target_grid_polar[0,0], target_grid_polar[matching_info.target_id_to_index[3],0]], [target_grid_polar[0,1], target_grid_polar[matching_info.target_id_to_index[3],1]], '--', color = '0.5')
		ax3.plot([target_grid_polar[0,0], target_grid_polar[matching_info.target_id_to_index[7],0]], [target_grid_polar[0,1], target_grid_polar[matching_info.target_id_to_index[7],1]], '--', color = '0.5')
	elif(analysis_params_general.center_type == 1):
		ax3.plot([target_grid_polar[6,0], target_grid_polar[matching_info.target_id_to_index[3],0]], [target_grid_polar[6,1], target_grid_polar[matching_info.target_id_to_index[3],1]], '--', color = '0.5')
		ax3.plot([target_grid_polar[6,0], target_grid_polar[matching_info.target_id_to_index[7],0]], [target_grid_polar[6,1], target_grid_polar[matching_info.target_id_to_index[7],1]], '--', color = '0.5')

	# set polar to pie
	ax3.set_thetamin(-50)
	ax3.set_thetamax(50)
	ax3.set_title('Mean Projection')

	#### color bar for polar plot
	vmin,vmax = sc.get_clim()
	cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)   #-- Defining a normalised scale
	ax4 = fig.add_axes([0.9, 0.2, 0.02, 0.6])       #-- Creating a new axes at the right side
	cb1 = matplotlib.colorbar.ColorbarBase(ax4, norm=cNorm, cmap=colormap)    #-- Plotting the colormap in the created axes
	fig.subplots_adjust(left=0.0,right=0.95)

	
	#### saving figure
	if(**kwarg['is_save']):
		plt.ioff()
		folder_name = img_name
		subfolder1 = 'PolarPlot'
		subfolder2 = f'slice_type_{slice_type}_analysis_params_general.center_type_{analysis_params_general.center_type}'
		# subfolder2 = 'slice_type_' + str(slice_type) + '_analysis_params_general.center_type' + str(analysis_params_general.center_type)
		# figure_name = 'bundle_no_' + str(bundle_no) + '_' + 'channel_' + matching_info.channel_mapping[channel_no] + '.tif'
		figure_name = f'bundle_no_{bundle_no}_channel_{matching_info.channel_mapping[channel_no]}.tif'

		my_help.check_dir(os.path.join(figure_prefix))
		my_help.check_dir(os.path.join(figure_prefix, folder_name))
		my_help.check_dir(os.path.join(figure_prefix, folder_name, subfolder1))
		my_help.check_dir(os.path.join(figure_prefix, folder_name, subfolder1, subfolder2))
		plt.savefig(os.path.join(figure_prefix, folder_name, subfolder1, subfolder2, figure_name), dpi=300, bbox_inches='tight')
	
	return fig



def plotMatrix(bundle_no, bundles_df, image, intensity_matrix, params, channel, figure_prefix, mind, is_plot_line, is_label_off, isSave, kwarg['is_true_x_tick']):
					   
	z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius = params
	
	phi_start = phis[0]
	phi_end = phis[-1]
					   
	### plot settings
	if(kwarg['is_true_x_tick']):
		x_ticks = np.linspace(0, 1, intensity_matrix.shape[1])
	else:
		x_ticks = np.arange(intensity_matrix.shape[1])
	
	if(channel == 'GFP'):
		colormap = 'Greens'
	elif(channel == 'RFP'):
		colormap = 'Reds'
	elif(channel == 'R4'):
		colormap = 'Greens'
	elif(channel == 'R3'):
		colormap = 'Reds'
	

	### R heels info
	r_heel_coords = np.zeros((6,2))
	r_heel_coords[:,0] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_X_R', True)])
	r_heel_coords[:,1] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_Y_R', True)])

	### targets info
	ind_targets, coord_targets = my_help.getTargetCoords(bundle_no, bundles_df)
	if(channel == 'GFP'):
		coord_rs = getR4Coords(bundle_no, bundles_df, ind_targets)
	elif(channel == 'R3'):
		coord_rs = getR3Coords(bundle_no, bundles_df, ind_targets)
	elif(channel == 'RFP'):
		coord_rs = getR4Coords(bundle_no, bundles_df, ind_targets)
	elif(channel == 'R4'):
		coord_rs = getR4Coords(bundle_no, bundles_df, ind_targets)
		
	coords = np.concatenate((coord_targets, coord_rs))

	### image boundary
	xlim = [np.floor(coords.min(axis = 0))[0] - 30, np.ceil(coords.max(axis = 0))[0] + 30]
	ylim = [np.floor(coords.min(axis = 0))[1] - 30, np.ceil(coords.max(axis = 0))[1] + 30]

	image_xy = image[r_z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), 0].shape
	
	### get image crops for plot
	bundle_img = exposure.rescale_intensity(image[r_z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), matching_info.channel_mapping[channel]], in_range = 'image', out_range='dtype')

	### plot
	fig = plt.figure(figsize = (18,8))
	fig.suptitle('Bundle No. ' + str(bundle_no), fontsize=18, fontweight='bold')



	### subfigure 2-4
	ax2 =  plt.subplot(231)
#     sns.heatmap(np.sum(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
	sns.heatmap(np.sum(intensity_matrix, axis = 2), cmap = colormap, yticklabels = False, xticklabels = False, vmin = 0, vmax = 20)
	ax2.set_ylabel('Angle', fontsize=16)
	ax2.set_xlabel('length from R-heel', fontsize=16)
	ax2.set_title('Sum over z-stack', fontsize=16)
	ax2.invert_yaxis()

	ax3 =  plt.subplot(232)
#     sns.heatmap(np.mean(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
	sns.heatmap(np.mean(intensity_matrix, axis = 2), cmap = colormap, yticklabels = False, xticklabels = False, vmin = 0, vmax = 20)
	ax3.set_ylabel('Angle', fontsize=16)
	ax3.set_xlabel('length from R-heel', fontsize=16)
	ax3.set_title('Mean intensity of z stack', fontsize=16)
	ax3.invert_yaxis()

	ax4 =  plt.subplot(233)
#     sns.heatmap(np.median(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
	sns.heatmap(np.median(intensity_matrix, axis = 2), cmap = colormap, yticklabels = False, xticklabels = False, vmin = 0, vmax = 20)
	ax4.set_ylabel('Angle', fontsize=16)
	ax4.set_xlabel('length from R-heel', fontsize=16)
	ax4.set_title('Median intensity of z stack', fontsize=16)
	ax4.invert_yaxis()


	ax5 =  plt.subplot(234)
#     sns.heatmap(np.max(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
	sns.heatmap(np.max(intensity_matrix, axis = 2), cmap = colormap, yticklabels = False, xticklabels = False, vmin = 0, vmax = 20)
	ax5.set_ylabel('Angle', fontsize=16)
	ax5.set_xlabel('length from R-heel', fontsize=16)
	ax5.set_title('Max intensity of z stack', fontsize=16)
	ax5.invert_yaxis()


	plt.subplots_adjust(wspace = 0.2, hspace=0.4)
	
	if(isSave):
		subfolder = 'bundles_' + channel
		figure_name = 'bundle_no_' + str(bundle_no) + '_' + channel + '_' + str(mind)
		plt.savefig(os.path.join(figure_prefix, subfolder, figure_name), dpi=300, bbox_inches='tight')


def plotSummaryMatrix(matrix, cmap, vmax, tick_params, labelParams):
	xTick_ori, yTick_ori, tick_type_x, tick_type_y, tick_arg2_x, tick_arg2_y = tick_params
	ylabel, xlabel, title = labelParams
	x_ticks = get_tick_list(tick_type_x, xTick_ori, tick_arg2_x)
	y_ticks = get_tick_list(tick_type_y, yTick_ori, tick_arg2_y)
	
	ax2 = plt.subplot(111)
	sns.heatmap(matrix, cmap = cmap, yticklabels = y_ticks, xticklabels = x_ticks, vmin = 0, vmax = vmax)
	ax2.set_ylabel(ylabel, fontsize=16)
	ax2.set_xlabel(xlabel, fontsize=16)
	ax2.set_title(title, fontsize=16)
	ax2.invert_yaxis()


# def plotBundleVsMatrix(bundle_no, bundles_df, image, intensity_matrix, fig_params, tick_params, plot_options, matching_info):
#     ## unravel parameters
#     channel, mind, params, figure_prefix, img_name, slice_type, radius_expanse_ratio = fig_params
#     is_plot_line, is_label_off, isSave, kwarg['is_true_x_tick'] = plot_options
#     tick_type_x, tick_type_y, tick_arg2_x, tick_arg2_y = tick_params
#     matching_info.index_to_target_id, matching_info.color_code, matching_info.channel_mapping, matching_info.target_id_to_index = matching_info

#     ## setting seaborn parameters
#     sns.set(font_scale=1.4)
#     sns.set_style("dark")
	
#     z_offset, num_x_section, r_z, phis, center_point, ori_y_ticks, radius = params
	
#     phi_start = phis[0]
#     phi_end = phis[-1]
					   
#     ### X and Y tick settings
#     if(kwarg['is_true_x_tick']):
#         ori_x_ticks = np.round(np.linspace(0, radius_expanse_ratio, intensity_matrix.shape[1]), 2)
# #         print(ori_x_ticks)
#         x_ticks = get_tick_list(tick_type_x, ori_x_ticks, tick_arg2_x)
#     else:
#         x_ticks = np.arange(intensity_matrix.shape[1])
#         # print(x_ticks)
	
#     y_ticks = get_tick_list(tick_type_y, ori_y_ticks, tick_arg2_y)
#     # print(y_ticks)
	
#     ### color code settings
#     if(channel == 'GFP'):
#         colormap = 'Greens'
#     elif(channel == 'RFP'):
#         colormap = 'Reds'
#     elif(channel == 'R4'):
#         colormap = 'Greens'
#     elif(channel == 'R3'):
#         colormap = 'Reds'
	

#     ### R heels info
#     r_heel_coords = np.zeros((6,2))
#     r_heel_coords[:,0] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_X_R', True)])
#     r_heel_coords[:,1] = list(bundles_df.loc[bundle_no, my_help.group_headers(bundles_df, 'coord_Y_R', True)])

#     ### targets info
#     ind_targets, coord_targets = my_help.getTargetCoords(bundle_no, bundles_df, matching_info.index_to_target_id)
#     if(channel == 'GFP'):
#         coord_rs = getR4Coords(bundle_no, bundles_df, ind_targets)
#     elif(channel == 'R3'):
#         coord_rs = getR3Coords(bundle_no, bundles_df, ind_targets)
#     elif(channel == 'RFP'):
#         coord_rs = getR4Coords(bundle_no, bundles_df, ind_targets)
#     elif(channel == 'R4'):
#         coord_rs = getR4Coords(bundle_no, bundles_df, ind_targets)
		
#     coords = np.concatenate((coord_targets, coord_rs))

#     ### image boundary
#     image_bd = (image.shape[2], image.shape[1])
#     xlim = [max(np.floor(coords.min(axis = 0))[0] - 21, 0), min(np.ceil(coords.max(axis = 0))[0] + 21, image_bd[0])]
#     ylim = [max(np.floor(coords.min(axis = 0))[1] - 21, 0), min(np.ceil(coords.max(axis = 0))[1] + 21, image_bd[1])]

#     image_xy = image[r_z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), 0].shape
	
#     ### get image crops for plot
#     bundle_img = exposure.rescale_intensity(image[r_z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), matching_info.channel_mapping[channel]], in_range = 'image', out_range='dtype')

#     ### plot
#     fig = plt.figure(figsize = (18,8))
#     fig.suptitle('Bundle No. ' + str(bundle_no) + '; Channel ' + channel, fontsize=18, fontweight='bold')
	
	
#     ### subplot 1: raw image
#     ax1 =  plt.subplot(231)
#     ## plot original bundle
#     for i in range(6):
#         ax1.plot(r_heel_coords[i,0]-xlim[0], r_heel_coords[i,1]-ylim[0], color = matching_info.color_code[i+1], marker = 'o', markersize=12)

#     ## plot targets
#     for i in matching_info.index_to_target_id.keys():
#         ax1.plot(coord_targets[i,0]-xlim[0], coord_targets[i,1]-ylim[0],'o', mec = matching_info.color_code[matching_info.index_to_target_id[i]], markersize=8, mew = 2.0, mfc = 'none')

#     ## plot R3/R4s
#     for i in range(1,len(coord_rs)):
#         ax1.plot(coord_rs[i,0]-xlim[0], coord_rs[i,1]-ylim[0], color = matching_info.color_code[4], marker = 'o', markersize=7)
	
#     ## plot lines
#     start_point = [ center_point[0] + radius * np.cos(phi_start) , center_point[1] + radius * np.sin(phi_start) ]
#     end_point = [ center_point[0] + radius * np.cos(phi_end) , center_point[1] + radius * np.sin(phi_end) ]
#     # Center-start
#     ax1.plot([ center_point[0] - xlim[0], start_point[0] - xlim[0] ], [ center_point[1] - ylim[0], start_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')
#     # Center-end
#     ax1.plot([ center_point[0] - xlim[0], end_point[0] - xlim[0] ], [ center_point[1] - ylim[0], end_point[1] - ylim[0] ], '--', alpha = 0.5, color = 'w')            
	
#     ## plot circle
#     circle = draw_circle(center_point, radius)
#     ax1.plot(circle[:,0] - xlim[0], circle[:,1] - ylim[0], '--', alpha = 0.5, color = 'w')

#     # plot image
#     ax1.imshow(bundle_img, cmap=plt.cm.gray)
	
#     # labels
#     ax1.set_title(channel + ' channel', fontsize=16)
	
# #     if(channel == 'GFP'):
# #         ax1.set_title('GFP channel', fontsize=16)
# #     elif(channel == 'R3'):
# #         ax1.set_title('RFP - GFP channel', fontsize=16)
# #     elif(channel == 'RFP'):
# #         ax1.set_title('RFP channel', fontsize=16)

#     if(is_label_off):
#         ax1.tick_params(axis='both', which='both', labelbottom='off', labelleft = 'off')
#     else:
#         ax1.tick_params(axis = 'both', labelsize = 12)


#     ### subfigure 2-4
#     ax2 =  plt.subplot(232)
#     sns.heatmap(np.sum(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
#     ax2.set_ylabel('Angle', fontsize=16)
#     ax2.set_xlabel('length from R-heel', fontsize=16)
#     ax2.set_title('Sum over z-stack', fontsize=16)
#     ax2.invert_yaxis()

#     ax3 =  plt.subplot(233)
#     sns.heatmap(np.mean(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
#     ax3.set_ylabel('Angle', fontsize=16)
#     ax3.set_xlabel('length from R-heel', fontsize=16)
#     ax3.set_title('Mean intensity of z stack', fontsize=16)
#     ax3.invert_yaxis()

#     ax4 =  plt.subplot(235)
#     sns.heatmap(np.median(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
#     ax4.set_ylabel('Angle', fontsize=16)
#     ax4.set_xlabel('length from R-heel', fontsize=16)
#     ax4.set_title('Median intensity of z stack', fontsize=16)
#     ax4.invert_yaxis()
	
#     ax5 =  plt.subplot(236)
#     sns.heatmap(np.max(intensity_matrix, axis = 2), cmap = colormap, yticklabels = y_ticks, xticklabels = x_ticks)
#     ax5.set_ylabel('Angle', fontsize=16)
#     ax5.set_xlabel('length from R-heel', fontsize=16)
#     ax5.set_title('Max intensity of z stack', fontsize=16)
#     ax5.invert_yaxis()
	
#     plt.subplots_adjust(wspace = 0.2, hspace=0.4)
	
#     if(isSave):
#         plt.ioff()
#         subfolder = 'slice_type' + str(slice_type) + '_' + channel
#         figure_name = 'bundle_no_' + str(bundle_no) + '_' + channel + '_' + str(mind)
		
#         my_help.check_dir(os.path.join(figure_prefix))
#         my_help.check_dir(os.path.join(figure_prefix, img_name))
#         my_help.check_dir(os.path.join(figure_prefix, img_name, subfolder))
		
#         plt.savefig(os.path.join(figure_prefix, img_name, subfolder, figure_name))
	
#     return fig

# def plotIndividualBundles_new(bundle_no, bundles_df, image_norm, xRatio, yRatio, plotSettings, matching_info):
#     ### Unravel plot settings
#     isPlotR3Line, isPlotR4Line, isPlotR4s, is_label_off, subplotShape, fsize = plotSettings
#     matching_info.index_to_target_id, matching_info.color_code, matching_info.channel_mapping, matching_info.target_id_to_index = matching_info
	
#     ### R heels info
#     R4_Z = int(bundles_df.loc[bundle_no,'coord_Z_R4']) - 1
#     r_heel_coords = get_all_rcell_coords(bundle_no, bundles_df)
	
#     ### targets info
#     ind_targets, coord_targets = my_plot.my_help.getTargetCoords(bundle_no, bundles_df, matching_info.index_to_target_id)
#     coordR4s = my_plot.my_help.getRxCoords(bundle_no, bundles_df, ind_targets, 4)
#     coords = np.concatenate((coord_targets, coordR4s))
#     print(coord_targets)

#     ### image boundary
#     image_bd = (image_norm.shape[2], image_norm.shape[1])
#     xlim = [max(np.floor(coords.min(axis = 0))[0] - 50, 0), min(np.ceil(coords.max(axis = 0))[0] + 50, image_bd[0])]
#     ylim = [max(np.floor(coords.min(axis = 0))[1] - 50, 0), min(np.ceil(coords.max(axis = 0))[1] + 50, image_bd[1])]
#     # print(xlim, ylim)
	
#     image_xy = image_norm[R4_Z, int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), 0].shape
	
#     max_projection_image = np.max(image_norm[(R4_Z - z_offset[0]):(R4_Z + z_offset[1]), int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1]), :], axis = 0)
	
#     ### get image crops for plot
#     bundle_img = np.empty(image_xy + (image_norm.shape[-1],), dtype=image_norm.dtype, order='C')
#     for i in range(image_norm.shape[-1]):
#         bundle_img[:,:,i] = exposure.rescale_intensity(max_projection_image[:,:, i], in_range = 'image', out_range='dtype')


#     ### plot
#     fig, axes = plt.subplots(subplotShape[0], subplotShape[1], figsize = fsize)
#     fig.suptitle('Bundle No. ' + str(bundle_no), fontsize=18, fontweight='bold')

#     #     plt_i = 0
#     for plt_i in range(image_norm.shape[-1]):
#         ax = axes.ravel()[plt_i]
#         # plot original bundle
#         for i in range(6):
#             ax.plot(r_heel_coords[i,0]-xlim[0], r_heel_coords[i,1]-ylim[0], color = matching_info.color_code[i+1], marker = 'o', markersize=12)

#         # plot targets
#         for i in matching_info.index_to_target_id.keys():
#             ax.plot(coord_targets[i,0]-xlim[0], coord_targets[i,1]-ylim[0],'o', mec = matching_info.color_code[matching_info.index_to_target_id[i]], markersize=8, mew = 2.0, mfc = 'none')

#         # plot R4s
#         if(isPlotR4s):
#             for i in range(1,len(coordR4s)):
#                 ax.plot(coordR4s[i,0]-xlim[0], coordR4s[i,1]-ylim[0], color = matching_info.color_code[4], marker = 'o', markersize=7)

#         # plot lines
#         if(isPlotR3Line):
#             # lines associated with R3 & R4
#             # R3-T3
#             ax.plot([r_heel_coords[2,0]-xlim[0], coord_targets[2,0]-xlim[0]], [r_heel_coords[2,1]-ylim[0], coord_targets[2,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[3])
#             # R3-T7
#             ax.plot([r_heel_coords[2,0]-xlim[0], coord_targets[5,0]-xlim[0]], [r_heel_coords[2,1]-ylim[0], coord_targets[5,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[3])
#         if(isPlotR4Line):
#             # R4-T3
#             ax.plot([r_heel_coords[3,0]-xlim[0], coord_targets[2,0]-xlim[0]], [r_heel_coords[3,1]-ylim[0], coord_targets[2,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[4])
#             # R4-T7
#             ax.plot([r_heel_coords[3,0]-xlim[0], coord_targets[5,0]-xlim[0]], [r_heel_coords[3,1]-ylim[0], coord_targets[5,1]-ylim[0]], '--', alpha = 0.5, color = matching_info.color_code[4])            

#         # plot image
#         ax.imshow(bundle_img[:,:,plt_i], cmap=plt.cm.gray)

#         # plot axis
#         ax.set_title(matching_info.channel_mapping[plt_i], fontsize=16)

#     #         plt_i += 1

#         if(is_label_off):
#             ax.tick_params(axis='both', which='both', labelbottom='off', labelleft = 'off')
#         else:
#             ax.tick_params(axis = 'both', labelsize = 12)

""" ========================= helper functions ========================="""
# def my_help.group_headers(df, header_tag, isContain):
#     '''What the function name says'''
#     if isContain:
#         return [col for col in df.columns.values if header_tag in col]
#     else:
#         return [col for col in df.columns.values if header_tag not in col]

# def my_help.getTargetCoords(bundle_no, bundles_df, matching_info.index_to_target_id):
#     ind_targets = []
#     coord_targets = np.zeros((len(matching_info.index_to_target_id),2))
	
#     coord_targets[0,:] = np.array([ bundles_df.loc[bundle_no,'coord_X_Center'], bundles_df.loc[bundle_no,'coord_Y_Center'] ])
#     ind_targets.append(bundle_no)
	
#     for i in range(1,len(coord_targets)):
#         ind_targets.append(int(bundles_df.loc[bundle_no,'TargetNo_T' + str(matching_info.index_to_target_id[i])]))
#         coord_targets[i,:] = np.array([ bundles_df.loc[ind_targets[i],'coord_X_Center'], bundles_df.loc[ind_targets[i],'coord_Y_Center'] ])

#     return ind_targets, coord_targets

# def my_help.getRxCoords(bundle_no, bundles_df, ind_targets, Rtype):
#     coordRxs = np.zeros((len(ind_targets), 2))
#     coordHeaders = ['coord_X_R' + str(Rtype), 'coord_Y_R' + str(Rtype)]
	
#     for i in range(len(ind_targets)):
#         coordRxs[i,:] = np.array([ bundles_df.loc[ ind_targets[i], coordHeaders[0]], bundles_df.loc[ ind_targets[i], coordHeaders[1]]])

#     return coordRxs

# def getR4Coords(bundle_no, bundles_df, ind_targets):
#     coordR4s = np.zeros((len(ind_targets), 2))

#     for i in range(len(ind_targets)):
#         coordR4s[i,:] = np.array([ bundles_df.loc[ ind_targets[i],'coord_X_R4' ], bundles_df.loc[ ind_targets[i],'coord_Y_R4' ]])

#     return coordR4s

# def getR3Coords(bundle_no, bundles_df, ind_targets):
#     coordR3s = np.zeros((len(ind_targets), 2))

#     for i in range(len(ind_targets)):
#         coordR3s[i,:] = np.array([ bundles_df.loc[ ind_targets[i],'coord_X_R3' ], bundles_df.loc[ ind_targets[i],'coord_Y_R3' ]])

#     return coordR3s

# def my_help.check_dir(path):
#     if not os.path.exists(path):
# #         print('Making dir!')
#         os.makedirs(path)
#     if not os.path.exists(path):
#         print('Fail to make directionary!')

