# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-03-27 15:07:30


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
	- isRadians: True/False
	Output: radians (or degree) of the inner angle
"""
def inner_angle(v1, v2, isRadians):
	
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	if isRadians:
		return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
	else:
		return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


### angle normalization
""" 
	Function: normalize angle to -pi ~ pi 
	Input: angle (in radians)
	Output: angle (in radians)
"""
def angle_normalization(angle):
	
	if(angle<-np.pi):
		angle = angle + 2*np.pi
	if(angle>np.pi):
		angle = angle - 2*np.pi
	return angle

### difference between two angles
"""
	source: https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
	Funtion: calcualte the smallest difference between two angles.
	Input: x,y -- angles (in radians)
	Output: angle (in radians)
"""
def smallest_angle(x, y):
	
	return min((2 * np.pi) - abs(x - y), abs(x - y))

# ================= Visualization functions =================
"""
Function: plog slicing of angles.
Input: 
- start, end: angles in radians
- phis: array of angles that should start from "start" and end at "end"
Output: figure
"""
def plot_angles(start, end, phis):
	pselect = list(range(phis.size))
	y_s = 1*np.tan(start)
	y_n = 1*np.tan(end)
	y_phis = 1*np.tan(phis[pselect])

	xs = np.ones((phis.size + 2))

	number = len(y_phis)
	cmap = plt.get_cmap('Wistia')
	colors = [cmap(i) for i in np.linspace(0, 1, number)]

	fig = plt.figure()
	for i, color in enumerate(colors, start=0):
		plt.plot([0,1], [0,y_phis[i]], color=color)
	# for i in y_phis:
	#     plt.plot(, '-')
	plt.plot([0,1], [0,y_s], '-', linewidth = 2.0, color = 'b')
	plt.plot([0,1], [0,y_n], '-', linewidth = 2.0, color = 'k')

	plt.show()

"""
Function: plog slicing of angles.
Input: 
- start, zero, end: angles in radians
- phis: array of angles that should start from "start", mid-point at "zero" and end at "end"
Output: figure
"""
def plot_angles_v2(start, zero, end, phis):
	pselect = list(range(phis.size))
	y_s = 1*np.tan(start)
	y_z = 1*np.tan(zero)
	y_n = 1*np.tan(end)
	y_phis = 1*np.tan(phis[pselect])
	# ys = np.concatenate((np.array([y_s, y_n]),y_phis))

	xs = np.ones((phis.size + 2))

	number = len(y_phis)
	cmap = plt.get_cmap('Wistia')
	colors = [cmap(i) for i in np.linspace(0, 1, number)]

	fig = plt.figure()
	for i, color in enumerate(colors, start=0):
		plt.plot([0,1], [0,y_phis[i]], color=color)
	# for i in y_phis:
	#     plt.plot(, '-')
	plt.plot([0,1], [0,y_s], '-', linewidth = 2.0, color = 'b')
	plt.plot([0,1], [0,y_z], '-', linewidth = 2.0, color = 'g')
	plt.plot([0,1], [0,y_n], '-', linewidth = 2.0, color = 'k')

	plt.show()


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
	if((-np.pi <= phi_end <= -0.5*np.pi) & (phi_end*phi_start < 0)):
		phi_end_transform = phi_end + 2*np.pi
		phis = np.linspace(phi_start, phi_end_transform, num_of_slices)
	elif((-np.pi <= phi_start <= -0.5*np.pi) & (phi_end*phi_start < 0)):
		phi_start_transform = phi_start + 2*np.pi
		phis = np.linspace(phi_end, phi_start_transform, num_of_slices)
	else:
		phis = np.linspace(phi_start, phi_end, num_of_slices)

	phis [phis>np.pi] = phis[phis>np.pi] - 2*np.pi
	phis [phis<-np.pi] = phis[phis<-np.pi] + 2*np.pi
	
	return phis

### Grid calculation
"""
Function: calculating grid's relative position
Input: target_coords, r3_coord, r4_coord, coord_center -- coordinates
Output: rel_points -- relative coordinates
- rel_points[0,0:5]: T0, T2, T3, T4, T5, T7
- rel_points[0,6]: center
- rel_points[0,7:8]: R3, R4
"""
def cal_grid_rel_position(target_coords, r3_coord, r4_coord, coord_center, center_point, r_unit):
	target_id_to_index = settings.matching_info.target_id_to_index

	rel_points = np.zeros((1,9))
	#T0_rel 
	rel_points[0,0] = np.linalg.norm( center_point - target_coords[target_id_to_index[0]] )/r_unit
	#T2_rel
	rel_points[0,1] = np.linalg.norm( center_point - target_coords[target_id_to_index[2]] )/r_unit
	#T3_rel 
	rel_points[0,2] = np.linalg.norm( center_point - target_coords[target_id_to_index[3]] )/r_unit
	#T4_rel
	rel_points[0,3] = np.linalg.norm( center_point - target_coords[target_id_to_index[4]] )/r_unit
	#T5_rel
	rel_points[0,4] = np.linalg.norm( center_point - target_coords[target_id_to_index[5]] )/r_unit
	#T7_rel 
	rel_points[0,5] = np.linalg.norm( center_point - target_coords[target_id_to_index[7]] )/r_unit
	#Center_rel 
	rel_points[0,6] = np.linalg.norm( center_point - coord_center )/r_unit
	#R3_rel 
	rel_points[0,7] = np.linalg.norm( center_point - r3_coord )/r_unit
	#R4_rel 
	rel_points[0,8] = np.linalg.norm( center_point - r4_coord )/r_unit

	return rel_points

### Angle normalization v1: T3 = 0, T7 = 1
"""
Function: calculate parameters necessary for image intensity transformation
Input:
- bundles_df: dataframe containing bundle information
- bundles_params: parameters regarding that bundle -- bundle_no, target_inds, target_coords, coord_center, slice_zero_point, slice_one_point, length_one_point, center_point, r_no
- **kwarg: is_print, is_plot
Output: 
- params: parameters to passed on for intensity calculation function -- z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius
- rel_points: relative coordinates for target positions and heel positions
"""
def get_slice_params_v1(bundles_df, bundle_params, **kwarg):
	### decomposite parameters.
	analysis_params_general = settings.analysis_params_general
	radius_expanse_ratio = analysis_params_general.radius_expanse_ratio[analysis_params_general.center_type]
	
	bundle_no, target_inds, target_coords, coord_center, slice_zero_point, slice_one_point, length_one_point, center_point, r_no = bundle_params

	### R heels info
	r_z = int(bundles_df.loc[bundle_no,'coord_Z_R' + str(r_no)]) - 1
	
	r3_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 3)[0,:]
	r4_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 4)[0,:]

	### slice radius calculation
	r_unit = np.linalg.norm( center_point - length_one_point )
	radius = r_unit * radius_expanse_ratio
	
	### calculating grid's relative position
	rel_points = cal_grid_rel_position(target_coords, r3_coord, r4_coord, coord_center, center_point, r_unit)

	### slice phis calculation
	ang_start2r = np.arctan2( slice_zero_point[1] - center_point[1], slice_zero_point[0] - center_point[0] )
	ang_end2r = np.arctan2( slice_one_point[1] - center_point[1], slice_one_point[0] - center_point[0])
	
	phi_range = inner_angle(slice_one_point - center_point, slice_zero_point - center_point, True)
	phi_unit = phi_range/analysis_params_general.num_angle_section
	if(kwarg['is_print']):
		print("ang_start2r: ")
		print(ang_start2r, ang_end2r)
	
	if(((-np.pi <= ang_start2r <= -0.5*np.pi) | (-np.pi <= ang_end2r <= -0.5*np.pi)) & (ang_start2r*ang_end2r < 1) ):
		if((-np.pi <= ang_start2r <= -0.5*np.pi) & (-np.pi <= ang_end2r <= -0.5*np.pi)):
			phi_start = min(ang_start2r, ang_end2r) - analysis_params_general.num_outside_angle * phi_unit
			phi_end = max(ang_start2r, ang_end2r) + analysis_params_general.num_outside_angle * phi_unit
		else:
			phi_start = max(ang_start2r, ang_end2r) - analysis_params_general.num_outside_angle * phi_unit
			phi_end = min(ang_start2r, ang_end2r) + analysis_params_general.num_outside_angle * phi_unit
	else:
		phi_start = min(ang_start2r, ang_end2r) - analysis_params_general.num_outside_angle * phi_unit
		phi_end = max(ang_start2r, ang_end2r) + analysis_params_general.num_outside_angle * phi_unit

	phi_start = angle_normalization(phi_start)
	phi_end = angle_normalization(phi_end)

	phis = get_phis(phi_start, phi_end, analysis_params_general.num_angle_section + analysis_params_general.num_outside_angle * 2 + 1)

	x = ang_start2r
	y = phis[0]
	z = phis[-1]
	if(smallest_angle(ang_start2r, phis[-1]) < smallest_angle(ang_start2r, phis[0])):
		phis = np.flip(phis, axis = 0)
	
	### printing/plotting
	if(kwarg['is_print']):
		print("final:")
		print(phis)
	
	if(kwarg['is_plot']):
		plot_angles(ang_start2r, ang_end2r, phis)
	
	### ticks for angle axis.
	y_ticks = np.linspace(- 1-analysis_params_general.num_outside_angle * (phi_unit/phi_range)*2, 1 + analysis_params_general.num_outside_angle * (phi_unit/phi_range)*2, analysis_params_general.num_angle_section + analysis_params_general.num_outside_angle*2 + 1)
	y_ticks = np.round(y_ticks, 2)
	y_ticks[y_ticks == -0] = 0
	
	### consolidating final params
	params = analysis_params_general.z_offset, analysis_params_general.num_x_section, r_z, phis, center_point, y_ticks, radius

	return params, rel_points


### Angle normalization v3: T3 = -1, T4 = 0, T7 = 1
"""
Function: calculate parameters necessary for image intensity transformation
Input:
- bundles_df: dataframe containing bundle information
- bundles_params: parameters regarding that bundle -- bundle_no, target_inds, target_coords, coord_center, slice_zero_point, slice_one_point, length_one_point, center_point, r_no
- **kwarg: is_print, is_plot
Output: 
- params: parameters to passed on for intensity calculation function -- z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius
- rel_points: relative coordinates for target positions and heel positions
"""
def get_slice_params_v3(bundles_df, bundle_params, **kwarg):
	### decomposite parameters.
	analysis_params_general = settings.analysis_params_general
	radius_expanse_ratio = analysis_params_general.radius_expanse_ratio[analysis_params_general.center_type]
	target_id_to_index = settings.matching_info.target_id_to_index

	bundle_no, target_inds, target_coords, coord_center, SliceNegOnePoint, slice_one_point, length_one_point, center_point, r_no = bundle_params

	angle_sel_num = analysis_params_general.num_angle_section / 2
	analysis_params_general.num_outside_angle = analysis_params_general.num_outside_angle
	
	### R heels info
	r_z = int(bundles_df.loc[bundle_no,'coord_Z_R' + str(r_no)]) - 1
	r3_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 3)[0,:]
	r4_coord = my_help.get_rx_coords(bundle_no, bundles_df, target_inds, 4)[0,:]

	### slice radius calculation
	r_unit = np.linalg.norm( center_point - length_one_point )
	radius = r_unit * radius_expanse_ratio
	
	### calculating grid's relative position
	rel_points = cal_grid_rel_position(target_coords, r3_coord, r4_coord, coord_center, center_point, r_unit)

	### slice phis calculation
	angNegOne2R = np.arctan2( SliceNegOnePoint[1] - center_point[1], SliceNegOnePoint[0] - center_point[0] )
	angZero = np.arctan2(target_coords[target_id_to_index[4]][1] - center_point[1], target_coords[target_id_to_index[4]][0] - center_point[0])
	angOne2R = np.arctan2(slice_one_point[1] - center_point[1], slice_one_point[0] - center_point[0])
	
	## T3' ~ middle (-1 ~ 0)
	phi_range_1 = inner_angle(SliceNegOnePoint - center_point, target_coords[3] - center_point, True)
	phi_unit_1 = phi_range_1/angle_sel_num
	
	if(kwarg['is_print']):
		print("angNegOne - angZero: ")
		print(angNegOne2R, angZero)

	if(((-np.pi <= angNegOne2R <= -0.5*np.pi) | (-np.pi <= angZero <= -0.5*np.pi)) & (angNegOne2R*angZero < 1) ):
		if((-np.pi <= angNegOne2R <= -0.5*np.pi) & (-np.pi <= angZero <= -0.5*np.pi)):
			phi_start_1 = min(angNegOne2R, angZero) - analysis_params_general.num_outside_angle * phi_unit_1
			phi_end_1 = max(angNegOne2R, angZero) + analysis_params_general.num_outside_angle * phi_unit_1
		else:
			phi_start_1 = max(angNegOne2R, angZero) - analysis_params_general.num_outside_angle * phi_unit_1
			phi_end_1 = min(angNegOne2R, angZero) + analysis_params_general.num_outside_angle * phi_unit_1
	else:
		phi_start_1 = min(angNegOne2R, angZero) - analysis_params_general.num_outside_angle * phi_unit_1
		phi_end_1 = max(angNegOne2R, angZero) + analysis_params_general.num_outside_angle * phi_unit_1

	phi_start_1 = angle_normalization(phi_start_1)
	phi_end_1 = angle_normalization(phi_end_1)

	phis_1 = get_phis(phi_start_1, phi_end_1, angle_sel_num + analysis_params_general.num_outside_angle*2 + 1)

	if(smallest_angle(angNegOne2R, phis_1[-1]) < smallest_angle(angNegOne2R, phis_1[0])):
		phis_1 = np.flip(phis_1, axis = 0)

	n_1 = int(angle_sel_num + analysis_params_general.num_outside_angle + 1)
	
	phis_1 = phis_1[0:n_1]
	phis_1[n_1-1] = angZero

	y_ticks_1 = np.linspace(-1 - analysis_params_general.num_outside_angle*phi_unit_1/phi_range_1, 0, angle_sel_num + analysis_params_general.num_outside_angle + 1)

	if(kwarg['is_print']):
		print("final:")
		print(phis_1, y_ticks_1)
		print(len(phis_1), len(y_ticks_1))

	if(kwarg['is_plot']):
		plot_angles(angNegOne2R, angZero, phis_1)

	
	## middle ~ T3 (0 ~ 1)
	phi_range_2 = inner_angle(slice_one_point - center_point, target_coords[3] - center_point, True)
	phi_unit_2 = phi_range_2/angle_sel_num

	if(kwarg['is_print']):
		print("angZero -angOne: ")
		print(angZero, angOne2R)

	if(((-np.pi <= angZero <= -0.5*np.pi) | (-np.pi <= angOne2R <= -0.5*np.pi)) & (angZero*angOne2R < 1) ):
		if((-np.pi <= angZero <= -0.5*np.pi) & (-np.pi <= angOne2R <= -0.5*np.pi)):
			phi_start_2 = min(angZero, angOne2R) - analysis_params_general.num_outside_angle * phi_unit_2
			phi_end_2 = max(angZero, angOne2R) + analysis_params_general.num_outside_angle * phi_unit_2
		else:
			phi_start_2 = max(angZero, angOne2R) - analysis_params_general.num_outside_angle * phi_unit_2
			phi_end_2 = min(angZero, angOne2R) + analysis_params_general.num_outside_angle * phi_unit_2
	else:
		phi_start_2 = min(angZero, angOne2R) - analysis_params_general.num_outside_angle * phi_unit_2
		phi_end_2 = max(angZero, angOne2R) + analysis_params_general.num_outside_angle * phi_unit_2

	phi_start_2 = angle_normalization(phi_start_2)
	phi_end_2 = angle_normalization(phi_end_2)

	phis_2 = get_phis(phi_start_2, phi_end_2, angle_sel_num + analysis_params_general.num_outside_angle*2 + 1)

	if(smallest_angle(angZero, phis_2[-1]) < smallest_angle(angZero, phis_2[0])):
		phis_2 = np.flip(phis_2, axis = 0)

	n_2 = int(analysis_params_general.num_outside_angle+1)
	phis_2 = phis_2[n_2:]

	y_ticks_2 = np.linspace(0, 1 + analysis_params_general.num_outside_angle*phi_unit_2/phi_range_2, angle_sel_num + analysis_params_general.num_outside_angle + 1)

	### printing/plotting
	if(kwarg['is_print']):
		print("final:")
		print(phis_2, y_ticks_2)
		print(len(phis_2), len(y_ticks_2))

	if(kwarg['is_plot']):
		plot_angles(angZero, angOne2R, phis_2)
	
	### combining phis
	phis = np.concatenate((phis_1,phis_2), axis = 0)
	y_ticks_2 = y_ticks_2[1:]
	y_ticks = np.concatenate((y_ticks_1,y_ticks_2), axis = 0)
	y_ticks = np.round(y_ticks,2)
	y_ticks[y_ticks == -0] = 0

	### printing/plotting
	if(kwarg['is_print']):
		print(phis, y_ticks)
		print(len(phis), len(y_ticks))

	if(kwarg['is_plot']):
		plot_angles_v2(angNegOne2R, angZero, angOne2R, phis)

	### consolidating final params	
	params = analysis_params_general.z_offset, analysis_params_general.num_x_section, r_z, phis, center_point, y_ticks, radius

	return params, rel_points

# ================= Calculate intensity matrix =================
### intensity matrix calculation
"""
Function: calculate intensity matrix for normalized grwoth cone.
Input:
- params: parameters generated from previous function -- z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius
- image: normalized image
- channel: int indicating which channel to calculate
Output: intensity_matrix
"""

def get_intensity_matrix_new(params, image):
	### decomposite parameters.
	z_max = image.shape[0]
	z_offset, num_x_section, r_z, phis, center_point, y_ticks, radius = params

	### initialization
	intensity_matrix = np.zeros((len(phis), num_x_section + 1, z_offset*2+1))
	intensity_matrix = intensity_matrix - 100
	z_values = np.linspace((r_z-z_offset), (r_z+z_offset), z_offset*2+1).astype(int)

	### get meshgrid of x and y inds.
	matrix_shape = image[0,:,:].shape
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

	### calculate intensity matrix
	# Error: too close to the boundary to generate full matrix
	if((np.max(vxf) > matrix_shape[1]) | (np.max(vyf) > matrix_shape[0]) | (np.min(vxf) < 0) | (np.min(vyf) < 0)):
		print_content = f'vx_max = {np.max(vxf)}, vy_max = {np.max(vyf)}; xboundary = {matrix_shape[1]}, yboundary = {matrix_shape[0]}'
		print("Error! Too close to the boundary!")
		print(print_content)

	else:
		for z in z_values:
			# correct situation.
			if((z >= 0) & (z < z_max)):
				gridded = interpolate.griddata(np.column_stack((vxf, vyf)), image[z,vy,vx].flatten(), (xs, ys), method='linear')
				intensity_matrix[:,:,z - r_z-z_offset] = gridded
			# Error: not enough z slices for calculation
			else:
				print("Error! not enough Z!")
		
	return intensity_matrix