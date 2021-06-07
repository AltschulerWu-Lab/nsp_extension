# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2021-06-06 17:55:00


import pandas as pd
import numpy as np
from scipy import interpolate

import helper as my_help
import settings as settings
import plotting as my_plot


# ================= angle calculation functions =================
### Inner angle calculation
# source: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
	""" 
	Function: Returns the unit vector of the vector.  
	Input: vector
	Output: vector
	"""
	return vector / np.linalg.norm(vector)

def inner_angle(v1, v2, is_radians):
	""" 
	Function: Returns the angle in radians(or degree) between vectors 'v1' and 'v2' 
	Input: 
	- v1/v2: vectors
	- is_radians: True/False
	Output: radians (or degree) of the inner angle
	"""
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	if is_radians:
		return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
	else:
		return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


### smallest difference between two angles
def smallest_angle(x, y):
	"""
	source: https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
	Funtion: calcualte the smallest difference between two angles.
	Input: x,y -- angles (in radians)
	Output: angle (in radians)
	"""
	return min((2 * np.pi) - abs(x - y), abs(x - y))


# ================= Angle normalization functions =================
### return lower/upper bound of target
def get_relative_axis(c0, c1, c2, type):
	"""
	Funtion: given the two boundaries of target, return which one is the lower-bound and which one is the upper-bound based on position of the center of bundle-of-interest.
	Inputs: 
	- c0: numpy array. coordinate of center of bundle-of-interest
	- c1, c2: coordinates of the two bounds of target.
	- type: string. "low" for lower-bound or "high" for upper-bound.
	Output: numpy array. coordiante of target bound.
	"""
	if(type == 'low'):
		if((np.linalg.norm(c0 - c1)) < (np.linalg.norm(c0 - c2))):
			return c1
		else:
			return c2
	elif(type == 'high'):
		if((np.linalg.norm(c0 - c1)) < (np.linalg.norm(c0 - c2))):
			return c2
		else:
			return c1

### angle normalization to -pi ~ pi
def angle_normalization(angles):
	""" 
	Function: normalize angle (or list of angles) to -pi ~ pi 
	Input: angle as float or numpy array (in radians)
	Output: angle as float or numpy array (in radians)
	"""
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

### get_counter_angle
def get_counter_angle(start, end, is_radians):
	"""
	Funtion: normalize angle between start and end to >= 0.
	Inputs: 
	- start, end -- angles (in radians)
	- is_radians: Boolean. True if return radians, False if return degrees.
	Output: angle (in radians or degrees)
	"""
	angle = end - start
	if(angle < 0):
		if(is_radians):
			angle = angle + np.pi*2
		else:
			angle = angle+360
	return angle


### Angle slicing
def get_phis(phi_start, phi_end, num_of_slices):
	"""
	Function: slice angle from "phi_start" to "phi_end" into equal slices (n = "num_of_slices")
	Input:
	- phi_start, phi_end: angles in radians
	- num_of_slices: int
	Output: phis -- array of angles in radians
	"""
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
def get_start_end(ang_start2r, ang_end2r, num_outside_angle, phi_unit):
	"""
	Function: expand pie range beyond ang_start2r - ang_end2r by num_outside_angle*phi_unit each.
	Input:
	- ang_start2r, ang_end2r(np.folat64): angles in radians
	- num_outside_angle (int)
	- phi_unit(float)
	Output: phis(np.ndarray) -- array of angles in radians
	"""
	counter_angle = get_counter_angle(ang_start2r, ang_end2r, True)
	if(counter_angle > np.pi):
		phi_start = angle_normalization(ang_start2r + num_outside_angle * phi_unit)
		phi_end = angle_normalization(ang_end2r - num_outside_angle * phi_unit)
	else:
		phi_start = angle_normalization(ang_start2r - num_outside_angle * phi_unit)
		phi_end = angle_normalization(ang_end2r + num_outside_angle * phi_unit)
	return phi_start, phi_end

### Grid calculation
def cal_grid_rel_position(coord_Tcs, coord_Tls, coord_Ths, r3_coord, r4_coord, heel_center, slice_center, r_unit):
	"""
	Function: calculating target grid's relative length.
	Input: target_coords, r3_coord, r4_coord, heel_center -- coordinates
	Output: 
	- rel_points -- relative lengths in dictionary
		keys = 'T0'-'T7', 'center', 'R3', 'R4'
	"""
	target_id_to_index = settings.matching_info.target_id_to_index

	# target_rel_poses
	rel_points = {}
	for i in [0,2,3,4,5,7]:
		rel_points[f'T{i}c'] = np.linalg.norm( slice_center - coord_Tcs[target_id_to_index[i]] )/r_unit
		rel_points[f'T{i}l'] = np.linalg.norm( slice_center - coord_Tls[target_id_to_index[i]] )/r_unit
		rel_points[f'T{i}h'] = np.linalg.norm( slice_center - coord_Ths[target_id_to_index[i]] )/r_unit
	# Center_rel 
	rel_points['center'] = np.linalg.norm( slice_center - heel_center )/r_unit
	# R3_rel 
	rel_points['R3'] = np.linalg.norm( slice_center - r3_coord )/r_unit
	# R4_rel 
	rel_points['R4'] = np.linalg.norm( slice_center - r4_coord )/r_unit

	return rel_points

### Angle normalization v1: T7 = -1, T3 = 1
def get_slice_params_v1(bundles_df, bundle_params, img_name, **kwargs):
	"""
	Function: calculate parameters necessary for image intensity transformation
	Input:
	- bundles_df: dataframe containing bundle information
	- bundles_params: parameters regarding that bundle -- bundle_no, target_inds, target_coords, heel_center, slice_neg_one_point, slice_one_point, length_one_point, slice_center, r_no
	- img_name: name of the plotted bundle slicing figure, if applicable.
	- kwargs: additional parameters.
		- is_print: Boolean. Whether to print moreinfo.
		- is_plot: Boolean. Whether to plot bundle slicing figure.
		- is_save: Boolean. whetehr to save the bundle slicing figure.
		- is_checking: Boolean. Determines which variables to return.
		- xy_ratio: float. pixel/um of the image.
	Outputs: 
	- is_checking == True:
		- phis: numpy array. angle value of each slice.
		- ang_start2r, ang_end2r: float. value of starting and ending angle of the pi.
		- rel_points: dictionary. relative length for target positions and heel positions

	- (is_checking == False) & (is_plot = True):
		- params: list. parameters to pass on for intensity calculation function -- z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius
		- rel_points: dictionary. relative length for target positions and heel positions
		- fig: figure to visualize angle slicing.
	- (is_checking == False) & (is_plot = False):
		- params: list. parameters to pass on for intensity calculation function -- z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius
		- rel_points: dictionary. relative length for target positions and heel positions
	"""

	### decomposite parameters.
	analysis_params_general = settings.analysis_params_general
	radius_expanse_ratio = analysis_params_general.radius_expanse_ratio
	
	bundle_no, target_inds, coord_Tcs, coord_Tls, coord_Ths, heel_center, slice_neg_one_point, slice_one_point, length_one_point, slice_center, r_no = bundle_params

	if('is_print' in kwargs.keys()):
		is_print = kwargs['is_print']
	else:
		is_print = False
	if('is_plot' in kwargs.keys()):
		is_plot = kwargs['is_plot']
	else:
		is_plot = kwargs['is_plot']
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('is_checking' in kwargs.keys()):
		is_checking = kwargs['is_checking']
	else:
		is_checking = False
	if('xy_ratio' in kwargs.keys()):
		xy_ratio = kwargs['xy_ratio']
	else:
		xy_ratio = 1

	### R heels info
	r_z = int(bundles_df.loc[bundle_no,'coord_Z_R' + str(r_no)]) - 1
	
	r3_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 3)[0,:]
	r4_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 4)[0,:]

	### slice radius calculation
	r_unit = np.linalg.norm( slice_center - length_one_point )
	radius = r_unit * radius_expanse_ratio
	
	### calculating grid's relative position
	rel_points = cal_grid_rel_position(coord_Tcs, coord_Tls, coord_Ths, r3_coord, r4_coord, heel_center, slice_center, r_unit)

	### slice phis calculation
	# -1: T7
	ang_start2r = np.arctan2( slice_neg_one_point[1] - slice_center[1], slice_neg_one_point[0] - slice_center[0] )
	# 1: T3
	ang_end2r = np.arctan2( slice_one_point[1] - slice_center[1], slice_one_point[0] - slice_center[0])
	# range and unit
	phi_range = inner_angle(slice_one_point - slice_center, slice_neg_one_point - slice_center, True)
	phi_unit = phi_range/analysis_params_general.num_angle_section
	
	phi_range_1 = inner_angle(slice_neg_one_point - slice_center, length_one_point - slice_center, True)
	phi_range_2 = inner_angle(slice_one_point - slice_center, length_one_point - slice_center, True)

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
	params = analysis_params_general.z_offset, analysis_params_general.num_x_section, r_z, phis, slice_center, y_ticks, radius

	rel_points['length_one_um'] = r_unit/xy_ratio
	rel_points['phi_range_1'] = phi_range_1
	rel_points['phi_range_2'] = phi_range_2
	
	if(is_checking):
		return phis, [ang_start2r, ang_end2r], rel_points
	else:
		if(is_plot):
			return params, rel_points, fig
		else:
			return params, rel_points

### Angle normalization v3: T7 = -1, T4 = 0, T3 = 1
def get_slice_params_v2(bundles_df, bundle_params, img_name, **kwargs):
	"""
	Function: calculate parameters necessary for image intensity transformation
	Input:
	- bundles_df: dataframe containing bundle information
	- bundles_params: parameters regarding that bundle -- bundle_no, target_inds, target_coords, heel_center, slice_neg_one_point, slice_one_point, length_one_point, slice_center, r_no
	- kwargs: additional parameters.
		- is_print: Boolean. Whether to print moreinfo.
		- is_plot: Boolean. Whether to plot bundle slicing figure.
		- is_save: Boolean. whetehr to save the bundle slicing figure.
		- is_checking: Boolean. Determines which variables to return.
		- xy_ratio: float. pixel/um of the image.
	Outputs: 
	- is_checking == True:
		- phis: angle value of each slice.
		- ang_start2r, ang_end2r: value of starting and ending angle of the pi.
		- rel_points: relative length for target positions and heel positions

	- (is_checking == False) & (is_plot = True):
		- params: parameters to pass on for intensity calculation function -- z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius
		- rel_points: relative length for target positions and heel positions
		- fig: figure to visualize angle slicing.
	- (is_checking == False) & (is_plot = False):
		- params: parameters to pass on for intensity calculation function -- z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius
		- rel_points: relative length for target positions and heel positions
	"""

	### decomposite parameters.
	analysis_params_general = settings.analysis_params_general
	radius_expanse_ratio = analysis_params_general.radius_expanse_ratio
	target_id_to_index = settings.matching_info.target_id_to_index

	bundle_no, target_inds, coord_Tcs, coord_Tls, coord_Ths, heel_center, slice_neg_one_point, slice_one_point, length_one_point, slice_center, r_no = bundle_params

	angle_sel_num = analysis_params_general.num_angle_section / 2
	analysis_params_general.num_outside_angle = analysis_params_general.num_outside_angle
	
	if('is_print' in kwargs.keys()):
		is_print = kwargs['is_print']
	else:
		is_print = False
	if('is_plot' in kwargs.keys()):
		is_plot = kwargs['is_plot']
	else:
		is_plot = kwargs['is_plot']
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('is_checking' in kwargs.keys()):
		is_checking = kwargs['is_checking']
	else:
		is_checking = False
	if('xy_ratio' in kwargs.keys()):
		xy_ratio = kwargs['xy_ratio']
	else:
		xy_ratio = 1

	### R heels info
	r_z = int(bundles_df.loc[bundle_no,'coord_Z_R' + str(r_no)]) - 1
	r3_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 3)[0,:]
	r4_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 4)[0,:]

	### slice radius calculation
	r_unit = np.linalg.norm( slice_center - length_one_point )
	radius = r_unit * radius_expanse_ratio
	
	### calculating grid's relative position
	rel_points = cal_grid_rel_position(coord_Tcs, coord_Tls, coord_Ths, r3_coord, r4_coord, heel_center, slice_center, r_unit)

	### slice phis calculation
	# -1: T7
	ang_negone2r = np.arctan2( slice_neg_one_point[1] - slice_center[1], slice_neg_one_point[0] - slice_center[0] )
	# 0: T4
	ang_zero = np.arctan2(length_one_point[1] - slice_center[1], length_one_point[0] - slice_center[0])
	# 1: T3
	ang_one2r = np.arctan2(slice_one_point[1] - slice_center[1], slice_one_point[0] - slice_center[0])
	
	## T3' ~ middle (-1 ~ 0)
	phi_range_1 = inner_angle(slice_neg_one_point - slice_center, length_one_point - slice_center, True)
	phi_unit_1 = phi_range_1/angle_sel_num
	
	phi_start, _ = get_start_end(ang_negone2r, ang_zero, analysis_params_general.num_outside_angle, phi_unit_1)

	phis_1 = get_phis(phi_start, ang_zero, angle_sel_num + analysis_params_general.num_outside_angle + 1)

	y_ticks_1 = np.linspace(-1 - analysis_params_general.num_outside_angle*phi_unit_1/phi_range_1, 0, int(angle_sel_num + analysis_params_general.num_outside_angle + 1))


	## middle ~ T3 (0 ~ 1)
	phi_range_2 = inner_angle(slice_one_point - slice_center, length_one_point - slice_center, True)
	phi_unit_2 = phi_range_2/angle_sel_num

	_, phi_end = get_start_end(ang_zero, ang_one2r, analysis_params_general.num_outside_angle, phi_unit_2)

	phis_2 = get_phis(ang_zero, phi_end, angle_sel_num + analysis_params_general.num_outside_angle + 1)

	y_ticks_2 = np.linspace(0, 1 + analysis_params_general.num_outside_angle*phi_unit_2/phi_range_2, int(angle_sel_num + analysis_params_general.num_outside_angle + 1))
	
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
	params = analysis_params_general.z_offset, analysis_params_general.num_x_section, r_z, phis, slice_center, y_ticks, radius

	rel_points['length_one_um'] = r_unit/xy_ratio
	rel_points['phi_range_1'] = phi_range_1
	rel_points['phi_range_2'] = phi_range_2

	if(is_checking):
		return phis, [ang_negone2r, ang_zero, ang_one2r], rel_points
	else:
		if(is_plot):
			return params, rel_points, fig
		else:
			return params, rel_points

### check if slicing exceeds image boundary.
def check_boundary(params, image):
	"""
	Function: check if slicing exceeds image boundary. If yes, give warning.
	Inputs:
	- params: list. parameters to pass on for intensity calculation function 	
		- z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius.
	- image: numpy array. intensity-scaled image matrix.
	Outputs:
	print warnings.
	"""
	### params
	z_max = image.shape[0]
	z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius = params

	intensity_matrix = np.zeros((len(phis), num_x_section + 1, z_offset*2+1))
	intensity_matrix = intensity_matrix - 100
	Z_values = np.linspace((r_z-z_offset), (r_z+z_offset), z_offset*2+1).astype(int)

	matrix_shape = image[0,:,:].shape


	### get vx and vy
	xs = np.zeros((len(phis), num_x_section + 1))
	ys = np.zeros((len(phis), num_x_section + 1))
	for phi in phis:
		i = int(np.argwhere(phis == phi))
		circleY = radius * np.sin(phi) + slice_center[1]
		circleX = radius * np.cos(phi) + slice_center[0]

		xs[i,:] = np.linspace(slice_center[0], circleX, num_x_section + 1)
		ys[i,:] = (xs[i,:] - slice_center[0]) * np.tan(phi) + slice_center[1]
	
	xbound = np.array(range(int(np.floor(np.min(xs))), int(np.ceil(np.max(xs)+1))))
	ybound = np.array(range(int(np.floor(np.min(ys))), int(np.ceil(np.max(ys)+1))))

	
	### determine if xbound and ybound is out of boundary and if yes, print warning.
	is_warning = False
	
	if(min(xbound) < 0):
		if not (is_warning):
			my_help.print_to_log("WARNING: ")
		my_help.print_to_log('xbound < 0! ')
		is_warning = True
	
	if(min(ybound) < 0):
		if not (is_warning):
			my_help.print_to_log("WARNING: ")
		my_help.print_to_log('ybound < 0! ')
		is_warning = True
	
	if(max(xbound) >= matrix_shape[1]):
		if not (is_warning):
			my_help.print_to_log("WARNING: ")
		my_help.print_to_log(f'xbound > {matrix_shape[1]}! ')
		is_warning = True
	
	if(max(ybound) >= matrix_shape[0]):
		if not (is_warning):
			my_help.print_to_log("WARNING: ")
		my_help.print_to_log(f'ybound > {matrix_shape[0]}! ')
		is_warning = True

### calculate standardized density map matrix.
def get_intensity_matrix_new(params, image):
	"""
	Function: calculate standardized density map matrix.
	Inputs:
	- params: list. parameters to pass on for intensity calculation function 	
		- z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius.
	- image: numpy array. intensity-scaled image matrix.
	Outputs: 
	- intensity_matrix: numpy array. standardized density map matrix.
	"""

	### params
	z_max = image.shape[0]
	z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius = params

	intensity_matrix = np.zeros((len(phis), num_x_section + 1, z_offset*2+1))
	intensity_matrix = intensity_matrix - 100
	Z_values = np.linspace((r_z-z_offset), (r_z+z_offset), z_offset*2+1).astype(int)

	matrix_shape = image[0,:,:].shape


	### get vx and vy
	xs = np.zeros((len(phis), num_x_section + 1))
	ys = np.zeros((len(phis), num_x_section + 1))
	for phi in phis:
		i = int(np.argwhere(phis == phi))
		circleY = radius * np.sin(phi) + slice_center[1]
		circleX = radius * np.cos(phi) + slice_center[0]

		xs[i,:] = np.linspace(slice_center[0], circleX, num_x_section + 1)
		ys[i,:] = (xs[i,:] - slice_center[0]) * np.tan(phi) + slice_center[1]
	
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
		ix_min = sum(xbound<0)
		shape_x += sum(xbound<0)
	if(min(ybound) < 0):
		iy_min = sum(ybound<0)
		shape_y += sum(ybound<0)
	if(max(xbound) >= matrix_shape[1]):
		shape_x += sum(xbound >= matrix_shape[1])
	if(max(ybound) >= matrix_shape[0]):
		shape_y += sum(ybound >= matrix_shape[0])

	ix_max = ix_min + matrix_shape[1]
	iy_max = iy_min + matrix_shape[0]

	vy = vy + iy_min
	vx = vx + ix_min
	xs = xs + ix_min
	ys = ys + iy_min
	vxf = vx.flatten()
	vyf = vy.flatten()
	
	### calculate gridded interpolation
	for z in Z_values:
		if((z >= 0) & (z < z_max)):
			image_matrix = image[z,:,:]
			### padding
			new_img_matrix = np.zeros((shape_y, shape_x))
			new_img_matrix[iy_min:iy_max,ix_min:ix_max] = image_matrix
			### grid
			gridded = interpolate.griddata(np.column_stack((vxf, vyf)), new_img_matrix[vy,vx].flatten(), (xs, ys), method='linear')
			intensity_matrix[:,:,z - r_z-z_offset] = gridded
		else:
			print("\nERROR! not enough Z!", end = "")
			my_help.print_to_log("\nERROR! not enough Z!")
	
	return intensity_matrix


### calculate boundary of image interpolation.
def calculate_boundary(params, img_shape):
	z_max = img_shape[0]
	z_offset, num_x_section, r_z, phis, slice_center, y_ticks, radius = params

	xs = np.zeros((len(phis), num_x_section + 1))
	ys = np.zeros((len(phis), num_x_section + 1))
	
	for phi in phis:
		i = int(np.argwhere(phis == phi))
		circleY = radius * np.sin(phi) + slice_center[1]
		circleX = radius * np.cos(phi) + slice_center[0]

		xs[i,:] = np.linspace(slice_center[0], circleX, num_x_section + 1)
		ys[i,:] = (xs[i,:] - slice_center[0]) * np.tan(phi) + slice_center[1]
	
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