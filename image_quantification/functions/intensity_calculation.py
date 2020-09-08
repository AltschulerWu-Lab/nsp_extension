# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-04-21 11:57:09


import io, os, sys, types

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import eig, inv

import math

from scipy import interpolate, spatial, stats

import seaborn as sns

import skimage.io as skiIo
from skimage import exposure, img_as_float

from sklearn import linear_model
from sklearn import metrics

# import cv2

import helper as my_help
import settings as settings
import plotting as my_plot


# ================= angle calculation functions =================
def get_relative_axis(c0, c1, c2):
	if((np.linalg.norm(c0 - c1)) < (np.linalg.norm(c0 - c2))):
		return c1
	else:
		return c2

# ================= angle calculation functions =================
### Inner angle calculation
# source: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
""" 
Function: Returns the unit vector of the vector.  
Input: vector
Output: vector
"""
def unit_vector(vector):
	
	return vector / np.linalg.norm(vector)

""" 
	Function: Returns the angle in radians(or degree) between vectors 'v1' and 'v2' 
	Input: 
	- v1/v2: vectors
	- is_radians: True/False
	Output: radians (or degree) of the inner angle
"""
def inner_angle(v1, v2, is_radians):
	
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	if is_radians:
		return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
	else:
		return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


### angle normalization
""" 
	Function: normalize angle (or list of angles) to -pi ~ pi 
	Input: angle as float or numpy array (in radians)
	Output: angle as float or numpy array (in radians)
"""
def angle_normalization(angles):
	if(np.isscalar(angles)):
		if(angles<-np.pi):
			angles = angles + 2*np.pi
		if(angles>np.pi):
			angles = angles - 2*np.pi
		return angles
	elif(type(angles) == np.ndarray):
		angles[angles>np.pi] = angles[angles>np.pi] - 2*np.pi
		angles[angles<-np.pi] = angles[angles<-np.pi] + 2*np.pi
		return angles
	else:
		print(f'{type(angles)} datatype not supported in angle_normalization!')
		return None

### difference between two angles
"""
	source: https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
	Funtion: calcualte the smallest difference between two angles.
	Input: x,y -- angles (in radians)
	Output: angle (in radians)
"""
def smallest_angle(x, y):
	
	return min((2 * np.pi) - abs(x - y), abs(x - y))


"""
"""
def get_counter_angle(start, end, is_radians):
	angle = end - start
	if(angle < 0):
		if(is_radians):
			angle = angle + np.pi*2
		else:
			angle = angle+360
	return angle

# ================= Angle normalization functions =================
### Angle slicing
"""
Function: slice angle from "phi_start" to "phi_end" into equal slices (n = "num_of_slices")
Input:
- phi_start, phi_end: angles in radians
- num_of_slices: int
Output: phis -- array of angles in radians
"""
def get_phis(phi_start, phi_end, num_of_slices):
	if(get_counter_angle(phi_start, phi_end, True) > np.pi):
		phi_start, phi_end = phi_end, phi_start
		is_flip = True
	else:
		is_flip = False
	if((-np.pi <= phi_end <= 0) & (phi_start*phi_end < 0)):
		phi_end = phi_end + 2*np.pi
	phis = np.linspace(phi_start, phi_end, int(num_of_slices))
	if(is_flip):
		phis = np.flip(phis, axis = 0)
	phis = angle_normalization(phis)
	return phis

### start and end angle calculation
"""
Function: expand pie range beyond ang_start2r-ang_end2r by num_outside_angle*phi_unit each.
Input:
- ang_start2r, ang_end2r(np.folat64): angles in radians
- num_outside_angle (int)
- phi_unit(float)
Output: phis(np.ndarray) -- array of angles in radians
"""
def get_start_end(ang_start2r, ang_end2r, num_outside_angle, phi_unit):
	counter_angle = get_counter_angle(ang_start2r, ang_end2r, True)
	if(counter_angle > np.pi):
		phi_start = angle_normalization(ang_start2r + num_outside_angle * phi_unit)
		phi_end = angle_normalization(ang_end2r - num_outside_angle * phi_unit)
	else:
		phi_start = angle_normalization(ang_start2r - num_outside_angle * phi_unit)
		phi_end = angle_normalization(ang_end2r + num_outside_angle * phi_unit)
	return phi_start, phi_end

### Grid calculation
"""
Function: calculating target grid's relative position
Input: target_coords, r3_coord, r4_coord, coord_center -- coordinates
Output: 
- rel_points -- relative coordinates in dictionary
	keys = 'T0'-'T7', 'center', 'R3', 'R4'
"""
def cal_grid_rel_position(target_coords, target_coords_extended, r3_coord, r4_coord, coord_center, center_point, r_unit):
	target_id_to_index = settings.matching_info.target_id_to_index

	# target_rel_poses
	rel_points = {}
	for i in [0,2,3,4,5,7]:
		rel_points[f'T{i}'] = np.linalg.norm( center_point - target_coords[target_id_to_index[i]] )/r_unit
		rel_points[f'T{i}_etd'] = np.linalg.norm( center_point - target_coords_extended[target_id_to_index[i]] )/r_unit
	#Center_rel 
	rel_points['center'] = np.linalg.norm( center_point - coord_center )/r_unit
	#R3_rel 
	rel_points['R3'] = np.linalg.norm( center_point - r3_coord )/r_unit
	#R4_rel 
	rel_points['R4'] = np.linalg.norm( center_point - r4_coord )/r_unit


	return rel_points

### Angle normalization v1: T7 = -1, T3 = 1
"""
Function: calculate parameters necessary for image intensity transformation
Input:
- bundles_df: dataframe containing bundle information
- bundles_params: parameters regarding that bundle -- bundle_no, target_inds, target_coords, coord_center, slice_neg_one_point, slice_one_point, length_one_point, center_point, r_no
- **kwarg: is_print, is_plot
Output: 
- params: parameters to passed on for intensity calculation function -- z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius
- rel_points: relative coordinates for target positions and heel positions
"""
def get_slice_params_v1(bundles_df, bundle_params, img_name, **kwarg):
	### decomposite parameters.
	analysis_params_general = settings.analysis_params_general
	radius_expanse_ratio = analysis_params_general.radius_expanse_ratio[analysis_params_general.center_type]
	
	bundle_no, target_inds, target_coords, target_coords_extended, coord_center, slice_neg_one_point, slice_one_point, length_one_point, center_point, r_no = bundle_params

	if('is_print' in kwarg.keys()):
		is_print = kwarg['is_print']
	else:
		is_print = False
	if('is_plot' in kwarg.keys()):
		is_plot = kwarg['is_plot']
	else:
		is_plot = kwarg['is_plot']
	if('is_save' in kwarg.keys()):
		is_save = kwarg['is_save']
	else:
		is_save = False
	if('is_checking' in kwarg.keys()):
		is_checking = kwarg['is_checking']
	else:
		is_checking = False

	### R heels info
	r_z = int(bundles_df.loc[bundle_no,'coord_Z_R' + str(r_no)]) - 1
	
	r3_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 3)[0,:]
	r4_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 4)[0,:]

	### slice radius calculation
	r_unit = np.linalg.norm( center_point - length_one_point )
	radius = r_unit * radius_expanse_ratio
	
	### calculating grid's relative position
	rel_points = cal_grid_rel_position(target_coords, target_coords_extended, r3_coord, r4_coord, coord_center, center_point, r_unit)

	### slice phis calculation
	# -1: T7
	ang_start2r = np.arctan2( slice_neg_one_point[1] - center_point[1], slice_neg_one_point[0] - center_point[0] )
	# 1: T3
	ang_end2r = np.arctan2( slice_one_point[1] - center_point[1], slice_one_point[0] - center_point[0])
	# range and unit
	phi_range = inner_angle(slice_one_point - center_point, slice_neg_one_point - center_point, True)
	phi_unit = phi_range/analysis_params_general.num_angle_section
	
	# start and end angle of pie slices.
	phi_start, phi_end = get_start_end(ang_start2r, ang_end2r, analysis_params_general.num_outside_angle, phi_unit)
		
	# get lists of angle slices.
	phis1 = get_phis(phi_start, ang_start2r, analysis_params_general.num_outside_angle+1)
	phis2 = get_phis(ang_start2r, ang_end2r, analysis_params_general.num_angle_section+1)
	phis3 = get_phis(ang_end2r, phi_end, analysis_params_general.num_outside_angle+1)
	phis = np.array(list(phis1) + list(phis2[1:-1]) + list(phis3))
	
	### printing/plotting
	if(is_print):
		print(f'ang_start2r={ang_start2r}, ang_end2r={ang_end2r}, phi_range={phi_range}, phi_unit={phi_unit}')
		print(f'phi_start={phi_start}, phi_end={phi_end}')
		print("final phis:")
		print(phis)
	if(is_plot):
		fig = my_plot.plot_angles(phis, [ang_start2r, ang_end2r], img_name, bundle_no, is_save = is_save)
	
	### ticks for angle axis.
	y_ticks = np.linspace(- 1-analysis_params_general.num_outside_angle * (phi_unit/phi_range)*2, 1 + analysis_params_general.num_outside_angle * (phi_unit/phi_range)*2, analysis_params_general.num_angle_section + analysis_params_general.num_outside_angle*2 + 1)
	y_ticks = np.round(y_ticks, 2)
	y_ticks[y_ticks == -0] = 0
	
	### consolidating final params
	params = analysis_params_general.z_offset, analysis_params_general.num_x_section, r_z, phis, center_point, y_ticks, radius
	
	if(is_checking):
		return phis, [ang_start2r, ang_end2r], rel_points
	else:
		if(is_plot):
			return params, rel_points, fig
		else:
			return params, rel_points

### Angle normalization v3: T7 = -1, T4 = 0, T3 = 1
"""
Function: calculate parameters necessary for image intensity transformation
Input:
- bundles_df: dataframe containing bundle information
- bundles_params: parameters regarding that bundle -- bundle_no, target_inds, target_coords, coord_center, slice_neg_one_point, slice_one_point, length_one_point, center_point, r_no
- **kwarg: is_print, is_plot
Output: 
- params: parameters to passed on for intensity calculation function -- z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius
- rel_points: relative coordinates for target positions and heel positions
"""
def get_slice_params_v3(bundles_df, bundle_params, img_name, **kwarg):
	### decomposite parameters.
	analysis_params_general = settings.analysis_params_general
	radius_expanse_ratio = analysis_params_general.radius_expanse_ratio[analysis_params_general.center_type]
	target_id_to_index = settings.matching_info.target_id_to_index

	bundle_no, target_inds, target_coords, target_coords_extended, coord_center, slice_neg_one_point, slice_one_point, length_one_point, center_point, r_no = bundle_params

	angle_sel_num = analysis_params_general.num_angle_section / 2
	analysis_params_general.num_outside_angle = analysis_params_general.num_outside_angle
	
	if('is_print' in kwarg.keys()):
		is_print = kwarg['is_print']
	else:
		is_print = False
	if('is_plot' in kwarg.keys()):
		is_plot = kwarg['is_plot']
	else:
		is_plot = kwarg['is_plot']
	if('is_save' in kwarg.keys()):
		is_save = kwarg['is_save']
	else:
		is_save = False
	if('is_checking' in kwarg.keys()):
		is_checking = kwarg['is_checking']
	else:
		is_checking = False

	### R heels info
	r_z = int(bundles_df.loc[bundle_no,'coord_Z_R' + str(r_no)]) - 1
	r3_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 3)[0,:]
	r4_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 4)[0,:]

	### slice radius calculation
	r_unit = np.linalg.norm( center_point - length_one_point )
	radius = r_unit * radius_expanse_ratio
	
	### calculating grid's relative position
	rel_points = cal_grid_rel_position(target_coords, target_coords_extended, r3_coord, r4_coord, coord_center, center_point, r_unit)

	### slice phis calculation
	# -1: T7
	ang_negone2r = np.arctan2( slice_neg_one_point[1] - center_point[1], slice_neg_one_point[0] - center_point[0] )
	# 0: T4
	ang_zero = np.arctan2(target_coords[target_id_to_index[4]][1] - center_point[1], target_coords[target_id_to_index[4]][0] - center_point[0])
	# 1: T3
	ang_one2r = np.arctan2(slice_one_point[1] - center_point[1], slice_one_point[0] - center_point[0])
	
	## T3' ~ middle (-1 ~ 0)
	phi_range_1 = inner_angle(slice_neg_one_point - center_point, target_coords[3] - center_point, True)
	phi_unit_1 = phi_range_1/angle_sel_num
	
	phi_start, _ = get_start_end(ang_negone2r, ang_zero, analysis_params_general.num_outside_angle, phi_unit_1)

	phis_1 = get_phis(phi_start, ang_zero, angle_sel_num + analysis_params_general.num_outside_angle + 1)

	y_ticks_1 = np.linspace(-1 - analysis_params_general.num_outside_angle*phi_unit_1/phi_range_1, 0, int(angle_sel_num + analysis_params_general.num_outside_angle + 1))


	## middle ~ T3 (0 ~ 1)
	phi_range_2 = inner_angle(slice_one_point - center_point, target_coords[3] - center_point, True)
	phi_unit_2 = phi_range_2/angle_sel_num

	_, phi_end = get_start_end(ang_zero, ang_one2r, analysis_params_general.num_outside_angle, phi_unit_2)

	phis_2 = get_phis(ang_zero, phi_end, angle_sel_num + analysis_params_general.num_outside_angle + 1)

	y_ticks_2 = np.linspace(0, 1 + analysis_params_general.num_outside_angle*phi_unit_2/phi_range_2, int(angle_sel_num + analysis_params_general.num_outside_angle + 1))

	# if(is_plot):
	#   plot_angles(ang_zero, ang_one2r, phis_2)
	
	### combining phis
	phis = np.array(list(phis_1) + list(phis_2[1:]))
	y_ticks_2 = y_ticks_2[1:]
	y_ticks = np.concatenate((y_ticks_1,y_ticks_2), axis = 0)
	y_ticks = np.round(y_ticks,2)
	y_ticks[y_ticks == -0] = 0

	### printing/plotting
	if(is_print):
		print(f'-1 = {ang_negone2r}, 0 = {ang_zero}, 1 = {ang_one2r}')
		print(f'phi_1({len(phis_1)}), y_ticks_1({len(y_ticks_1)}):')
		print(phis_1, y_ticks_1)
		print(f'phi_2({len(phis_2)}), y_ticks_2({len(y_ticks_2)}):')
		print(phis_2, y_ticks_2)
		print(f'final phis({len(phis)}), y_ticks({len(y_ticks)}):')
		print(phis, y_ticks)

	if(is_plot):
		fig = my_plot.plot_angles(phis, [ang_negone2r, ang_zero, ang_one2r], img_name, bundle_no, is_save = is_save)

	### consolidating final params  
	params = analysis_params_general.z_offset, analysis_params_general.num_x_section, r_z, phis, center_point, y_ticks, radius

	if(is_checking):
		return phis, [ang_negone2r, ang_zero, ang_one2r], rel_points
	else:
		if(is_plot):
			return params, rel_points, fig
		else:
			return params, rel_points

def get_intensity_matrix_old(params, image):
	### decomposite parameters.
	# analysis_params_general = settings.analysis_params_general
	# radius_expanse_ratio = analysis_params_general.radius_expanse_ratio[analysis_params_general.center_type]
	# target_id_to_index = settings.matching_info.target_id_to_index
	# channel_mapping = settings.matching_info.channel_mapping


	z_max = image.shape[0]
	z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius = params

	intensity_matrix = np.zeros((len(phis), num_x_section + 1, z_offset*2+1))
	intensity_matrix = intensity_matrix - 100
	Z_values = np.linspace((r_z-z_offset), (r_z+z_offset), z_offset*2+1).astype(int)


	matrix_shape = image[0,:,:,].shape
	xs = np.zeros((len(phis), num_x_section + 1))
	ys = np.zeros((len(phis), num_x_section + 1))
	for phi in phis:
		i = int(np.argwhere(phis == phi))
		circleY = radius * np.sin(phi) + center_point[1]
		circleX = radius * np.cos(phi) + center_point[0]

		xs[i,:] = np.linspace(center_point[0], circleX, num_x_section + 1)
		ys[i,:] = (xs[i,:] - center_point[0]) * np.tan(phi) + center_point[1]
	
	xbound = np.array(range(int(np.floor(np.min(xs))), int(np.ceil(np.max(xs)+1))))
	ybound = np.array(range(int(np.floor(np.min(ys))), int(np.ceil(np.max(ys)+1))))

	vy, vx = np.meshgrid(ybound, xbound)

	vxf = vx.flatten()
	vyf = vy.flatten()


	if((np.max(vxf) > matrix_shape[1]) | (np.max(vyf) > matrix_shape[0]) | (np.min(vxf) < 0) | (np.min(vyf) < 0)):
		print_content = f'vx_max = {np.max(vxf)}, vy_max = {np.max(vyf)}; xboundary = {matrix_shape[1]}, yboundary = {matrix_shape[0]}'
		print("ERROR! Too close to the boundary:", end = " ")
		print(print_content)
		my_help.print_to_log("ERROR! Too close to the boundary: ")
		my_help.print_to_log(print_content)
		my_help.print_to_log("\n")
	else:
		for z in Z_values:
			if((z >= 0) & (z < z_max)):
				imageMatrix = image[z,:,:]
				matrix_shape = imageMatrix.shape
				gridded = interpolate.griddata(np.column_stack((vxf, vyf)), imageMatrix[vy,vx].flatten(), (xs, ys), method='linear')
				intensity_matrix[:,:,z - r_z-z_offset] = gridded
			else:
				print("ERROR! not enough Z!")
				my_help.print_to_log("ERROR! not enough Z!\n")
		
	return intensity_matrix

def get_intensity_matrix_new(params, image):

	z_max = image.shape[0]
	z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius = params

	intensity_matrix = np.zeros((len(phis), num_x_section + 1, z_offset*2+1))
	intensity_matrix = intensity_matrix - 100
	Z_values = np.linspace((r_z-z_offset), (r_z+z_offset), z_offset*2+1).astype(int)


	matrix_shape = image[0,:,:,].shape

	### get vx and vy
	xs = np.zeros((len(phis), num_x_section + 1))
	ys = np.zeros((len(phis), num_x_section + 1))
	for phi in phis:
		i = int(np.argwhere(phis == phi))
		circleY = radius * np.sin(phi) + center_point[1]
		circleX = radius * np.cos(phi) + center_point[0]

		xs[i,:] = np.linspace(center_point[0], circleX, num_x_section + 1)
		ys[i,:] = (xs[i,:] - center_point[0]) * np.tan(phi) + center_point[1]
	
	xbound = np.array(range(int(np.floor(np.min(xs))), int(np.ceil(np.max(xs)+1))))
	ybound = np.array(range(int(np.floor(np.min(ys))), int(np.ceil(np.max(ys)+1))))

	vy, vx = np.meshgrid(ybound, xbound)

	vxf = vx.flatten()
	vyf = vy.flatten()

	
	### determine if xbound and ybound is out of boundary and if yes, pad the boundary of image with 0.
	ix_min = 0
	iy_min = 0
	shape_x = matrix_shape[1]
	shape_y = matrix_shape[0]
	if(min(xbound) < 0):
		print('\nWARNING! xbound<0!')
		my_help.print_to_log('\nWARNING! xbound<0!\n')
		ix_min = sum(xbound<0)
		shape_x += sum(xbound<0)
	if(min(ybound) < 0):
		print('\nWARNING! ybound<0!')
		my_help.print_to_log('\nWARNING! ybound<0!\n')
		iy_min = sum(ybound<0)
		shape_y += sum(ybound<0)
	if(max(xbound) >= matrix_shape[1]):
		print(f'\nWARNING! xbound>{matrix_shape[1]}!')
		my_help.print_to_log(f'\nWARNING! xbound>{matrix_shape[1]}!\n')
		shape_x += sum(xbound >= matrix_shape[1])
	if(max(ybound) >= matrix_shape[0]):
		print(f'\nWARNING! ybound>{matrix_shape[0]}!')
		my_help.print_to_log(f'\nWARNING! ybound>{matrix_shape[0]}!\n')
		shape_y += sum(ybound >= matrix_shape[0])

	ix_max = ix_min + matrix_shape[1]
	iy_max = iy_min + matrix_shape[0]

	vy = vy + iy_min
	vx = vx + ix_min


	### calculate gridded interpolation
	for z in Z_values:
		# print(f'{z}-', end = "")
		if((z >= 0) & (z < z_max)):
			imageMatrix = image[z,:,:]
			### padding
			new_img_matrix = np.zeros((shape_y, shape_x))
			new_img_matrix[iy_min:iy_max,ix_min:ix_max] = imageMatrix
			matrix_shape = imageMatrix.shape
			### grid
			gridded = interpolate.griddata(np.column_stack((vxf, vyf)), new_img_matrix[vy,vx].flatten(), (xs, ys), method='linear')
			intensity_matrix[:,:,z - r_z-z_offset] = gridded
		else:
			print("ERROR! not enough Z!")
			my_help.print_to_log("ERROR! not enough Z!\n")
	
	return intensity_matrix


def calculate_boundary(params, img_shape):
	z_max = img_shape[0]
	z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius = params

	xs = np.zeros((len(phis), num_x_section + 1))
	ys = np.zeros((len(phis), num_x_section + 1))
	
	for phi in phis:
		i = int(np.argwhere(phis == phi))
		circleY = radius * np.sin(phi) + center_point[1]
		circleX = radius * np.cos(phi) + center_point[0]

		xs[i,:] = np.linspace(center_point[0], circleX, num_x_section + 1)
		ys[i,:] = (xs[i,:] - center_point[0]) * np.tan(phi) + center_point[1]
	
	xbound = np.array(range(int(np.floor(np.min(xs))), int(np.ceil(np.max(xs)+1))))
	ybound = np.array(range(int(np.floor(np.min(ys))), int(np.ceil(np.max(ys)+1))))

	vy, vx = np.meshgrid(ybound, xbound)

	vxf = vx.flatten()
	vyf = vy.flatten()

	matrix_shape = img_shape[1:]
	if((np.max(vxf) > matrix_shape[1])):
		print("ERROR! vx_max out of boundary:", end = " ")
		print(f'vx_max = {np.max(vxf)}, xboundary = {matrix_shape[1]}')
	elif((np.min(vxf) < 0)):
		print("ERROR! vx_min < 0")
	elif((np.max(vyf) > matrix_shape[0])):
		print("ERROR! vy_max out of boundary:", end = " ")
		print(f'vy_max = {np.max(vyf)}, yboundary = {matrix_shape[0]}')
	elif((np.min(vyf) < 0)):
		print("ERROR! vy_min < 0")