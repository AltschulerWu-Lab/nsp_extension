# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2020-09-09 04:01:25
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2021-10-10 21:49:31

import os, matplotlib, math

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import numpy as np
import numpy.ma as ma

import scipy.stats as ss
import statsmodels.api as sa
import scikit_posthocs as sp

from sklearn import linear_model

import settings as settings

# ================= files =================
### get the foldres and files within a particular path
def parse_folder_info(path):
	"""
	Function: get the foldres and files within a particular path.
	Input: path
	Output: lists of folders and files
	"""

	folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	if('.DS_Store' in files):
		files.remove('.DS_Store')
	if('._.DS_Store' in files):
		files.remove('._.DS_Store')
	return folders, files

### sort dictionary according to keys or values
def sort_dic(dic, switch, is_reverse):
	"""
	Function: sort dictionary according to keys or values.
	Input: 
	- dic: Dictionary.
	- switch: str. "keys" or "values" to sort.
	- is_reverse: whether or not to sort reversely.
	Output: Dictionary. sorted.
	"""
	if(switch == 'keys'):
		return {k: v for k, v in sorted(dic.items(), key=lambda item: item[0], reverse = is_reverse)}
	elif(switch == 'values'):
		return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse = is_reverse)}

### group DataFrame columns: get DataFrame column names that have the same pattern
def group_headers(df, header_tag, isContain):
	'''
	Function: get DataFrame column names that have the same pattern (contain or doesn't contain a particular string)
	Input: 
	- df -- dataframe
	- header_tag -- string
	- isContain -- True/False
	Output: list of strings
	'''

	if isContain:
		return [col for col in df.columns.values if header_tag in col]
	else:
		return [col for col in df.columns.values if header_tag not in col]

### add columns to DataFrame
def dataframe_add_column(df, column_list):
	'''
	Function: add columns to a DataFrame.
	Input: 
	- df: DataFrame.
	- column_list: columns to add.
	Output: df (w/ new columns)
	'''
	### check if columns in column_list exist in dataframe already
	new_col = []
	for col in column_list:
		if(col not in df.columns):
			new_col.append(col)

	### if not, append.
	if(len(new_col) > 0):
		df = df.reindex( columns = df.columns.tolist() + new_col )

	return df


# ================= geometry =================
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

### smallest difference between two angles
def smallest_angle(x, y):
	"""
	source: https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
	Funtion: calcualte the smallest difference between two angles.
	Input: x,y -- angles (in radians)
	Output: angle (in radians)
	"""
	return min((2 * np.pi) - abs(x - y), abs(x - y))

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

### get_vector_length
def get_vector_length(v):
	"""
	Function: get length of a vector.
	Input: numpy array. vector.
	Output: float. length of the vector
	"""
	return np.linalg.norm(v)

### Calculate line intersections
def line_intersection(line1, line2):
	"""
	Function: Calculate intersection between two lines.
	Source: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
	Input: line1 - (point1, point2); line2 - (point1, point2)
	Output: x,y - floats. x and y coordinates of intersection.
	"""

	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
	   raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y



# ================= reformatting grid =================
### polar-cartasian conversion
def cartesian_to_polar(x,y):
	r = np.sqrt(x**2 + y**2)
	theta = np.arctan(y/x)
	return r,theta

def polar_to_cartesian(r,theta):
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	return x,y


### re-center coordinates
def carcedian_re_center_coords(coords, center):
	"""
	Function: re-center array of coordinates according to the center coordinate
	Inputs:
	- coords: numpy array. array of coordinates (can either be nx2 or 2xn)
	- center: numpy array. coordinate of the center (1x2 or 2x1)
	Outputs:
	- new_coords: numpy array. array of coordinates re-centered. same format as coords.
	"""
	new_coords = np.copy(coords)
	shape = coords.shape
	if(shape[0] == 2):
		new_coords[0,:] = new_coords[0,:] - center[0]
		new_coords[1,:] = new_coords[1,:] - center[1]
	elif(shape[1] == 2):
		new_coords[:,0] = new_coords[:,0] - center[0]
		new_coords[:,1] = new_coords[:,1] - center[1]
	return new_coords

### flip coordinates horizontally or vertically.
def flip_coords(coords, axis):
	"""
	Function: flip coordinates horizontally or vertically
	Inputs:
	- coords: numpy array. array of coordinates (can either be nx2 or 2xn)
	- axis: str. 'v' = flip vertically; 'h' = flip horizontally.
	Outputs:
	- new_coords: numpy array. array of coordinates re-centered. same format as coords.
	"""
	new_coords = np.copy(coords)
	shape = coords.shape
	if(axis == 'h'):
		if(shape[0] == 2):
			new_coords[0,:] = - new_coords[0,:]
		elif(shape[1] == 2):
			 new_coords[:,0] = - new_coords[:,0]
	if(axis == 'v'):
		if(shape[0] == 2):
			new_coords[1,:] = - new_coords[1,:]
		elif(shape[1] == 2):
			 new_coords[:,1] = - new_coords[:,1]
	return new_coords

### rotate coordinates counter-clockwise.
def rotate_points(center_point, coords, angle):
	"""
	Function: Rotates coordinates counter-clockwise around a center point. Rotation angle is in radians.
	Source: adapted from https://gist.github.com/somada141/d81a05f172bb2df26a2c
	Input:
	- center_point: numpy array. 1x2 or 2x1. 
	- coords: numpy array. array of coordinates (nx2).
	- angle: float. rotation angle in radians.
	Output:
	- new_coords: numpy array (nx2). new coordinates after rotation.
	"""
	new_coords = np.zeros(coords.shape)
	new_coords[:,0] = coords[:,0] - center_point[0]
	new_coords[:,1] = coords[:,1] - center_point[1]
	
	new_coords[:,0] = new_coords[:,0] * math.cos(angle) - new_coords[:,1] * math.sin(angle)
	new_coords[:,1] = new_coords[:,0] * math.sin(angle) + new_coords[:,1] * math.cos(angle)
	
	new_coords[:,0] = new_coords[:,0] + center_point[0]
	new_coords[:,1] = new_coords[:,1] - center_point[1]
	
	return new_coords

### get centroids of given coordinates.
def get_centroid(coords):
	"""
	Function: get centroids of given coordinates.
	Input:
	- coords: numpy array. mx2xn. m = number of centroids; n = number of points per centroid.
	Output:
	- new_coords: numpy array (mx2). centroids.
	"""
	new_coords = np.zeros((coords.shape[0], coords.shape[1]))
	for i in range(coords.shape[0]):
		new_coords[i,0] = np.sum(coords[i,0,:])/coords.shape[2]
		new_coords[i,1] = np.sum(coords[i,1,:])/coords.shape[2]
	return new_coords



### get heel coordinates
def get_heel_coords_sum(bundle_no, annots_df, **kwargs):
	"""
	Function: get coordinate of heel positions of a given bundle
	Inputs:
	- bundle_no: numpy array. array of coordinates (can either be nx2 or 2xn)
	- bundles_df: DataFrame. containing informations of bundles.
	- kwargs: additional parameters
		- dim: int. dimention of returning coordinates. 2 or 3.
		- is_pixel: Boolean. whether or not return coordinates in pixel (True) or um (False)
		- pixel_to_um: numpy array (1x2 or 1x3). um/pixel for each dimension.
	Outputs:
	- heel_coords: numpy array. array of heel coordinates.
	"""

	### unravel params
	dim = 2

	### get heel coordinates
	heel_coords = np.zeros((6,dim))
	heel_coords[:,0] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_X_R', True)])
	heel_coords[:,1] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_Y_R', True)])

	return heel_coords

### get target coordinates
def get_target_coords_sum(bundle_no, annots_df, **kwargs):
	"""
	Function: get coordinate of heel positions of a given bundle
	Inputs:
	- bundle_no: numpy array. array of coordinates (can either be nx2 or 2xn)
	- bundles_df: DataFrame. containing informations of bundles.
	- kwargs: additional parameters
		- dim: int. dimention of returning coordinates. 2 or 3.
		- is_pixel: Boolean. whether or not return coordinates in pixel (True) or um (False)
		- pixel_to_um: numpy array (1x2 or 1x3). um/pixel for each dimension.
	Outputs:
	- heel_coords: numpy array. array of heel coordinates.
	"""

	### unravel params
	dim = 2
	index_to_target_id = settings.matching_info.index_to_target_id

	### get target coordinates
	target_coords = np.zeros((6,dim))
	target_coords[:,0] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_X_T', True)])
	target_coords[:,1] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_Y_T', True)])
	
	return target_coords

### get angle unit information from theoretical grid.
def get_angle_unit_theory(return_type):
	"""
	Function: get angle unit information from theoretical grid.
	Input:
	- return_type: str. 
		- phi_unit: return radian value of the unit of standardized angle.
		- aTiCT4: return radian value of angles between targets, center, and T4.
		- aRiCT4: return radian value of angles between heels, center, and T4.
	Outputs:
	- phi_unit: float. radian value of the unit of standardized angle.
	- aTiCT4: numpy array (6x1). radian value of angles between targets, center, and T4.
	- aRiCT4: numpy array (6x1). radian value of angles between heels, center, and T4.
	"""


	### before standardization
	#### distance: normal
	dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
	dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
	dT2T5 = dT3T7 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
	dT0T3 = dT0T7 = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5

	#### angles: normal
	aT0T2 = math.radians(80)/2
	aT0T5 = - math.radians(80)/2
	aT0T3 = math.acos((dT0T3 ** 2 + dT0T7 ** 2 - dT3T7 ** 2)/(2*dT0T3*dT0T7))/2
	aT0T7 = - aT0T3
	aT0T4 = 0

	#### target coordinates
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

	#### heel coordinates
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

	### intersect
	c = line_intersection((r_heels_cart[2,:], target_grid_cart[2,:]),(r_heels_cart[3,:], target_grid_cart[5,:]))

	### after standardization
	dTiC = np.zeros((6,1))
	for i in range(1,6):
		dTiC[i] = np.linalg.norm(target_grid_cart[i,:] - c)
	dTiC = dTiC/dTiC[3]

	aTiCT4 = np.zeros((6,1))
	for i in range(1,6):
		aTiCT4[i] = inner_angle(target_grid_cart[i,:] - c, target_grid_cart[3,:] - c, True)
		if(i in [4,5]):
			aTiCT4[i] = - aTiCT4[i]

	aRiCT4 = np.zeros((6,1))
	for i in range(1,6):
		aRiCT4[i] = inner_angle(r_heels_cart[i,:] - c, target_grid_cart[3,:] - c, True)
		if(i in [4,5]):
			aRiCT4[i] = - aRiCT4[i]
	
	### phi_unit
	phi_unit = aTiCT4[2,0]


	### return
	if(return_type == 'phi_unit'):
		return phi_unit
	elif(return_type == 'aTiCT4'):
		return aTiCT4
	elif(return_type == 'aRiCT4'):
		return aRiCT4

### get angle unit information from measured target positions.
def get_angle_unit_data(sum_df, **kwargs):
	"""
	Function: get angle unit information from measured target positions.
	Input:
	- sum_df: DataFrame. processed DataFrame that contains both bundle heel and target info and growth cone length and angle info.
	- kwargs: additional parameters
		- 'criteria': Dataframe with Boolean values. filtering which bundles to include in the calculation.
	Output:
	- phi_unit: radian value of "1" in standardized coordinate.
	"""

	if('criteria' in kwargs.keys()):
		criteria = kwargs['criteria']
		sum_df = sum_df.loc[criteria, :]

	# print(f"get_angle_unit_num={len(sum_df)}")
	phi_unit = sum_df['aT3cT7'].mean()/2

	return phi_unit

### get polar coordiantes of target grid from standardized coordinates.
def get_target_grid_polar_summary(**kwargs):
	"""
	Function: get polar coordiantes of target grid.
	Input:
	- kwargs:
		- 'return_type': str. calculate angle based on theoretical grid ("theory") or measured grid ("data")
		- 'dTiCs': dictionary. {target_id : distance value}. |Ti-C| normalized.
		- 'aTiCT4s': numpy array. radian values of angles between Ti, C, T4.
	Output:
	- grid: numpy array (6x2). polar coordinate values of target grid (T0, T2, T3, T4, T5, T3')
	"""

	### unravel params
	index_to_target_id = settings.matching_info.index_to_target_id

	return_type = kwargs['return_type']
	if(return_type == 'theory'):
		dTiCs = kwargs['dTiCs']
		aTiCT4s = get_angle_unit_theory('aTiCT4')
	elif(return_type == 'data'):
		dTiCs = kwargs['dTiCs']
		aTiCT4s =  kwargs['aTiCT4s']

	### get grid
	grid = np.zeros((6,2))
	for i in range(6):
		#### theta
		grid[i,0] = aTiCT4s[i]
		#### r
		if(index_to_target_id[i] in dTiCs.keys()):
			grid[i,1] = dTiCs[index_to_target_id[i]]

	return grid

### get polar coordiantes of heel grid from standardized coordinates.
def get_heel_grid_polar_summary(**kwargs):
	"""
	Function: get polar coordiantes of target grid.
	Input:
	- kwargs:
		- 'return_type': str. calculate angle based on theoretical grid ("theory") or measured grid ("data")
		- 'dTiCs': dictionary. {target_id : distance value}. |Ti-C| normalized.
		- 'aTiCT4s': numpy array. radian values of angles between Ti, C, T4.
	Output:
	- grid: numpy array (6x2). polar coordinate values of target grid (T0, T2, T3, T4, T5, T3')
	"""

	### unravel parameters
	index_to_target_id = settings.matching_info.index_to_target_id

	return_type = kwargs['return_type']
	if(return_type == 'theory'):
		dRiCs = kwargs['dRiCs']
		aRiCT4 = get_angle_unit_theory('aRiCT4')
	elif(return_type == 'data'):
		dRiCs = kwargs['dRiCs']
		aRiCT4 =  kwargs['aRiCT4']

	### get grid info.
	grid = np.zeros((6,2))
	for i in range(6):
		grid[i,0] = aRiCT4[i]
		if(i+1 in dRiCs.keys()):
			grid[i,1] = dRiCs[i+1]

	return grid

### Standardized coordinate --> grid in cartasian coordinates.
def get_cartasian_grid_from_stc(sum_df_ri, ch, cat_angle, cat_length, phi_unit):
	### params
	target_id_to_index = settings.matching_info.target_id_to_index

	### target grid polar
	dTiCs = {
		3:sum_df_ri['T3c'], 
		7:sum_df_ri['T7c'],
		2:sum_df_ri['T2c'],
		5:sum_df_ri['T5c'],
		4:1,
	}
	aTiCT4s = np.zeros((6))
	aTiCT4s[target_id_to_index[3]] = phi_unit
	aTiCT4s[target_id_to_index[7]] = - phi_unit
	aTiCT4s[target_id_to_index[2]] = (sum_df_ri['aT2cT4']/sum_df_ri['aT3cT4'])*phi_unit
	aTiCT4s[target_id_to_index[5]] = -(sum_df_ri['aT5cT4']/sum_df_ri['aT7cT4'])*phi_unit
	target_stc_polar = get_target_grid_polar_summary(return_type = 'data', dTiCs = dTiCs, aTiCT4s = aTiCT4s)

	### heel grid polar
	dRiCs = {}
	aRiCT4s = np.zeros((6))
	for i in [1,2,5,6]:
		dRiCs[i] = get_vector_length(ch[i-1,:])/sum_df_ri['length_one_um']
		if(i in [1,2]):
			aRiCT4s[i-1] = inner_angle(ch[i-1,:], np.array([1,0]), True)/sum_df_ri['aT3cT4'] * phi_unit
		elif(i in [5,6]):
			aRiCT4s[i-1] = - inner_angle(ch[i-1,:], np.array([1,0]), True)/sum_df_ri['aT7cT4'] * phi_unit
	dRiCs[3] = sum_df_ri['R3']
	dRiCs[4] = sum_df_ri['R4']
	aRiCT4s[3-1] = target_stc_polar[2,0]
	aRiCT4s[4-1] = target_stc_polar[5,0]

	heels_stc_polar = get_heel_grid_polar_summary(return_type = 'data', dRiCs = dRiCs, aRiCT4 = aRiCT4s)

	### growth cone tipe polar
	gc_tip_polar = np.zeros((1,2))
	gc_tip_polar[0,0] = sum_df_ri[cat_angle] * phi_unit
	gc_tip_polar[0,1] = sum_df_ri[cat_length]

	
	### polar to cartasian
	target_stc_car = np.zeros((6,2))
	heels_stc_car = np.zeros((6,2))
	gc_tip_car = np.zeros((1,2))
	for i in range(6):
		heels_stc_car[i,0], heels_stc_car[i,1] = polar_to_cartesian(heels_stc_polar[i,1], heels_stc_polar[i,0])
		target_stc_car[i,0], target_stc_car[i,1] = polar_to_cartesian(target_stc_polar[i,1], target_stc_polar[i,0])
	gc_tip_car[0,0], gc_tip_car[0,1] = polar_to_cartesian(gc_tip_polar[0,1], gc_tip_polar[0,0])
		
	return target_stc_car, heels_stc_car, gc_tip_car

### get angle and length of growth cones.
def get_gc_angle_length(sum_df_ri, coord_heels, phi_unit, cat_angle, cat_length, r_type):
	### from standardized coordinate to cartasian coordinate
	target_stc_car, heels_stc_car, gc_tip_car = get_cartasian_grid_from_stc(sum_df_ri, coord_heels, cat_angle, cat_length, phi_unit)

	### get vector of growth cone extension
	if(r_type == 3):
		ori = heels_stc_car[2,:]
	else:
		ori = heels_stc_car[3,:]
	
	v_gc = gc_tip_car - ori

	### relative angle
	gc_angle = inner_angle(v_gc, np.array([1,0]), True)
	gc_angle_rel = gc_angle/phi_unit

	if(v_gc[0,1] < 0):
		gc_angle_rel = - gc_angle_rel

	### relative length.
	gc_lengtrh = get_vector_length(v_gc)

	return gc_lengtrh, gc_angle_rel



# ================= mutual repulsion calculation =================
### new vector based on two base vectors and its weights (alphas)
def get_angle_prediction_two_vectors(v1, v2, origin, alphas):
	v1_uni = unit_vector(v1)
	v2_uni = unit_vector(v2)
	
	v_new = alphas[0] * v1_uni + alphas[1] * v2_uni
	point = origin + v_new
	
	v_new = unit_vector(v_new)
	point = origin + v_new
	
	return point, v_new

### calculate theoretical angle
def calculate_mutual_repulsion_theory(coord_heels, coord_target, r_type):
	r_type = int(r_type)
	
	### params and initialization
	target_id_to_index = settings.matching_info.target_id_to_index
	
	### basics
	ori = coord_heels[r_type-1, :]
	
	if(r_type == 3):
		v1 = ori - coord_heels[2 -1, :]
		v2 = ori - coord_heels[4 -1, :]
		v_base = coord_heels[4-1,:] - coord_heels[3-1,:]
	elif(r_type == 4):
		v1 = ori - coord_heels[5 -1, :]
		v2 = ori - coord_heels[3 -1, :]
		v_base = coord_heels[3-1,:] - coord_heels[4-1,:]

	ls = np.zeros((2))
	ls[0] = get_vector_length(v1)
	ls[1] = get_vector_length(v2)

	# print(f"v1={v1}, v2={v2}.")


	### repulse from neighbor heels, weighted equally
	alpha = 0.5
	p, v = get_angle_prediction_two_vectors(v1, v2, ori, [alpha, 1-alpha])

	# print(f"p={p}, v = {v}")

	point = np.transpose(p)
	vector = np.transpose(v)
	theta = inner_angle(vector, v_base, True)
	angle = inner_angle(v, np.array([1,0]), True)

	return point, vector, theta, angle, np.vstack((v1, v2))

### calculate actual angle.
def calculate_mutual_repulsion_data(sum_df_ri, ch, phi_unit, cat_angle, cat_length, r_type):
	target_stc_car, heels_stc_car, gc_tip_car = get_cartasian_grid_from_stc(sum_df_ri, ch, cat_angle, cat_length, phi_unit)

	if(r_type == 3):
		gc_vector = gc_tip_car[0,:] - heels_stc_car[2,:]

		gc_theta = inner_angle(heels_stc_car[3,:] - heels_stc_car[2,:], gc_vector, True)
		
	elif(r_type == 4):
		gc_vector = gc_tip_car[0,:] - heels_stc_car[3,:]
		gc_theta = inner_angle(heels_stc_car[2,:] - heels_stc_car[3,:], gc_vector, True)
		
	
	gc_angle = inner_angle(gc_vector, np.array([1,0]), True)

	return gc_tip_car, gc_vector, gc_theta, gc_angle

### data for regression.
def generate_regression_data(sum_df):
	X = np.zeros((len(sum_df) * 2, 2))
	y = np.zeros((len(sum_df) * 2))    
	
	for i,ind in enumerate(sum_df.index):
		v1 = np.array([sum_df.loc[ind, 'ml_x_v1'], sum_df.loc[ind, 'ml_y_v1']])
		v2 = np.array([sum_df.loc[ind, 'ml_x_v2'], sum_df.loc[ind, 'ml_y_v2']])
		vy = np.array([sum_df.loc[ind, 'ml_x_vgc'], sum_df.loc[ind, 'ml_y_vgc']])

		v1_uni = unit_vector(v1)
		v2_uni = unit_vector(v2)
		vy_uni = unit_vector(vy)

		X[2*i+0, 0] = v1_uni[0]
		X[2*i+0, 1] = v2_uni[0]
		y[2*i+0] = vy_uni[0]

		X[2*i+1, 0] = v1_uni[1]
		X[2*i+1, 1] = v2_uni[1]
		y[2*i+1] = vy_uni[1]
	
	return X,y

### regression analysis for mutual repulsion
def mutual_repulsion_regression(sum_df, annots_df):
	### parameters
	paths = settings.paths
	
	### regression fitting
	criteria = (sum_df['symmetry']<=0.5) & (sum_df['time_id']<=26)
	sum_df_regression = sum_df.loc[criteria,:]
	print(len(sum_df_regression))
	df_regression_results = pd.DataFrame(columns = ['a', 'b', 'r2'])
	print("Regression result:")
	for i, r_type in enumerate(["R3", "R4"]):
		sum_df_r = sum_df_regression.groupby("type_plot").get_group(r_type)
		df_data = sum_df_r[['ml_x_v1', 'ml_y_v1', 'ml_x_v2', 'ml_y_v2', 'ml_x_vgc', 'ml_y_vgc']].dropna()
		X, y = generate_regression_data(df_data)
		model = linear_model.LassoCV(alphas=np.logspace(-6, -3, 7),
                     max_iter=100000,
                     cv=5,
                     fit_intercept=False,
                     positive=True)
		reg = model.fit(X,y)
		
		print(f"r_type = {r_type}: alpha = {reg.coef_[0]:.2f}, beta = {reg.coef_[1]:.2f}, R^2 = {reg.score(X,y):.2f}")

		df_tmp = pd.DataFrame(columns = df_regression_results.columns)
		df_tmp.loc[0, 'type_plot'] = r_type
		df_tmp.loc[0, 'a'] = reg.coef_[0]
		df_tmp.loc[0, 'b'] = reg.coef_[1]
		df_tmp.loc[0, 'r2'] = reg.score(X,y)
		df_regression_results = df_regression_results.append(df_tmp, ignore_index=True)
	
	### calculate regression direction
	sum_df_ctrl_group = sum_df_regression.groupby(["time_id", "sample_no"])
	phi_unit = get_angle_unit_data(annots_df, 
											  criteria = (annots_df['is_Edge'] == 0) & (annots_df['symmetry'] <= 0.5))
	print("Regression direction calculation:", end = " ")
	for gp in sum_df_ctrl_group.groups.keys():
		time_id, sample_id = gp
		print(f"{time_id}_hrs_sample_{sample_id}", end = "; ")

		sum_df_current = sum_df_ctrl_group.get_group(gp)
		annots_df_current = annots_df.groupby(["time_id", "sample_no"]).get_group(gp).set_index('bundle_no')

		for ind in sum_df_current.index:
			r_type = int(sum_df_current.loc[ind, 'type_Rcell'])
			bundle_no = sum_df_current.loc[ind,'bundle_no']

			coord_heels = get_heel_coords_sum(bundle_no, annots_df_current)

			ori = coord_heels[r_type-1, :]

			if(r_type == 3):
				v_base = coord_heels[4-1,:] - coord_heels[3-1,:]
			elif(r_type == 4):
				v_base = coord_heels[3-1,:] - coord_heels[4-1,:]

			type_plot = sum_df_current.loc[ind, 'type_plot']
			i_reg = df_regression_results['type_plot'] == type_plot
			alphas = np.zeros((2))
			alphas[0] = df_regression_results.loc[i_reg, 'a'].values[0]
			alphas[1] = df_regression_results.loc[i_reg, 'b'].values[0]
			
			v1 = np.array((sum_df_current.loc[ind, 'ml_x_v1'], sum_df_current.loc[ind, 'ml_y_v1']))
			v2 = np.array((sum_df_current.loc[ind, 'ml_x_v2'], sum_df_current.loc[ind, 'ml_y_v2']))
			_, v_pred = get_angle_prediction_two_vectors(v1, v2, ori, alphas)

			theta = inner_angle(v_base, v_pred, True)
			angle = inner_angle(np.array([1,0]), v_pred, True)

			sum_df.loc[ind, 'ml_theory_theta_reg'] = theta
			sum_df.loc[ind, 'ml_theory_angle_reg'] = angle
			sum_df.loc[ind, 'ml_theory_vec_x_reg'] = v_pred[0]
			sum_df.loc[ind, 'ml_theory_vec_y_reg'] = v_pred[1]

	for plot_cat in ['angle', 'theta']:
		theory_cat = f"ml_theory_{plot_cat}"
		actual_cat = f"ml_actual_{plot_cat}"
		sum_df[f"ml_diff_{plot_cat}"] = (sum_df[theory_cat] - sum_df[actual_cat])
		
		theory_cat = f"ml_theory_{plot_cat}_reg"
		actual_cat = f"ml_actual_{plot_cat}"
		sum_df[f"ml_diff_{plot_cat}_reg"] = (sum_df[theory_cat] - sum_df[actual_cat])

	return df_data

# ================= process annots_df ================= #
### process annotation files.
def process_annots_df(annots_df, rel_poses):
	"""
	Function: processing Dataframe with heel/target coordinates of bundles.
	
	Inputs:
	- annots_df: DataFrame. Imported bundle information csv.
	- rel_poses: Dictionaries. Relative position info from the image quantification process.
	
	Output:
	- annots_df: DataFrame. Processed DataFrame that combines relative position info and heel/target coordinates (center, orientation, and axis aligned).
	"""


	paths = settings.paths
	target_id_to_index = settings.matching_info.target_id_to_index
	index_to_target_id = settings.matching_info.index_to_target_id
	
	annots_df_group = annots_df.groupby(['time_id', 'sample_no'])
	### process individual time and sample
	for gp in annots_df_group.groups.keys():
		time_id, sample_id = gp
		print(f'{time_id}, {sample_id}; ', end = "")
		
		rel_pos = rel_poses[gp]
		annot_bundles_df = annots_df_group.get_group(gp).reset_index().set_index('bundle_no')
		annot_bundles_df.sort_index(inplace = True)

		### align target and heel positions.
		for i_bd, bundle_no in enumerate(annot_bundles_df.index):
			
			ind_annot = annot_bundles_df.loc[bundle_no, 'index']
			orientation = annot_bundles_df.loc[bundle_no, ['Orientation_AP', 'Orientation_DV']]

			### original target and heel coordinates.
			ct_ori = get_target_coords_sum(bundle_no, annot_bundles_df)
			ch_ori = get_heel_coords_sum(bundle_no, annot_bundles_df)
			center = line_intersection((ch_ori[2,:], ct_ori[target_id_to_index[3],:]),
											   (ch_ori[3,:], ct_ori[target_id_to_index[7],:]))
			center = np.array(center)


			### new coordinate initialization
			ct_new = carcedian_re_center_coords(ct_ori, center)
			ch_new = carcedian_re_center_coords(ch_ori, center)


			### flip coordinates so that heels are at same orientation.
			if(orientation['Orientation_AP'] != "A"):
				ct_new = flip_coords(ct_new, 'v')
				ch_new = flip_coords(ch_new, 'v')
			if(orientation['Orientation_DV'] != "R"):
				ct_new = flip_coords(ct_new, 'h')
				ch_new = flip_coords(ch_new, 'h')


			### rotate coordinates so that center-T4 line is x-axis.
			angle = inner_angle(np.array([1,0]) - np.array([0,0]), ct_new[3,:] - np.array([0,0]), True)
			if(ct_new[3,1] > 0):
				angle = 2*np.pi - angle
			ch_new = rotate_points(np.array([0,0]), ch_new, angle)
			ct_new = rotate_points(np.array([0,0]), ct_new, angle)


			### update the new coordinates to annots_df.
			for i in range(ch_new.shape[0]):
				annots_df.loc[ind_annot, f'coord_X_R{i+1}'] = ch_new[i,0]
				annots_df.loc[ind_annot, f'coord_Y_R{i+1}'] = ch_new[i,1]
				annots_df.loc[ind_annot, f'coord_X_T{index_to_target_id[i]}'] = ct_new[i,0]
				annots_df.loc[ind_annot, f'coord_Y_T{index_to_target_id[i]}'] = ct_new[i,1]

			### update other information to annots_df.
			phi_range_1 = rel_pos[bundle_no]["phi_range_1"]
			phi_range_2 = rel_pos[bundle_no]["phi_range_2"]
			symmetry = abs(phi_range_1 - phi_range_2)/max(phi_range_2, phi_range_1)
			annots_df.loc[ind_annot, 'symmetry'] = symmetry
			annots_df.loc[ind_annot, 'aT7cT4'] = rel_pos[bundle_no]['phi_range_1']
			annots_df.loc[ind_annot, 'aT3cT4'] = rel_pos[bundle_no]['phi_range_2']
			annots_df.loc[ind_annot, 'aT3cT7'] = phi_range_1 + phi_range_2
			annots_df.loc[ind_annot, 'aT2cT4'] = inner_angle(ct_new[target_id_to_index[2],:], ct_new[target_id_to_index[4],:], True)
			annots_df.loc[ind_annot, 'aT5cT4'] = inner_angle(ct_new[target_id_to_index[5],:], ct_new[target_id_to_index[4],:], True)
			annots_df.loc[ind_annot, 'aT2cT5'] = inner_angle(ct_new[target_id_to_index[2],:], ct_new[target_id_to_index[5],:], True)

			annots_df.loc[ind_annot, 'R3'] = rel_pos[bundle_no]["R3"]
			annots_df.loc[ind_annot, 'R4'] = rel_pos[bundle_no]["R4"]
			annots_df.loc[ind_annot, 'length_one_um'] = rel_pos[bundle_no]["length_one_um"]
			annots_df.loc[ind_annot, 'T3c'] = rel_pos[bundle_no]["T3c"]
			annots_df.loc[ind_annot, 'T7c'] = rel_pos[bundle_no]["T7c"]

	print("")
	return annots_df

# ================= process summary_df =================
### supporting function: fill sum_df information for each bundle.
def fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR, r_type, phi_unit_real, phi_unit_theory):
	"""
	Function: fill sum_df information for each bundle
	Inputs:
	- sum_df: DataFrame. summary of length/angle and annotations.
	- annots_df_current: DataFrame. Annotation csv.
	- rel_pos: Dictionary. relative lengths of targets and heels.
	- num_rcells: number of R cells per bundle.
	- bundle_no: Bundle No. of bundle-of-interest
	- iR: int. index of sum_df for this R-cell
	- r_type: int. type of R-cell (3 for R3 and 4 for R4.)
	- phi_unit: converting from standardized coordiante to polar coordinate.
	"""
	qc_cols = group_headers(annots_df_current, 'is', True)
	annot_angle_cols = group_headers(annots_df_current, 'aT', True)


	phi_range_1 = rel_pos[bundle_no]["phi_range_1"]
	phi_range_2 = rel_pos[bundle_no]["phi_range_2"]
	aT30T7 = phi_range_1 + phi_range_2
	symmetry = abs(phi_range_1 - phi_range_2)/max(phi_range_2, phi_range_1)
	coord_heels = get_heel_coords_sum(bundle_no, annots_df_current)
	# coord_targets = get_target_coords_sum(bundle_no, annots_df_current)

	### convert R4 angle to mirror-symmetric
	if(r_type == 4):
		sum_df.loc[iR, f'angle_mrr'] = 0 - sum_df.loc[iR, 'angle']
	elif(r_type == 3):
		sum_df.loc[iR, f'angle_mrr'] = sum_df.loc[iR, 'angle']

	### add total number of r cells in bundle
	sum_df.loc[iR,'bundle_rcells_total'] = num_rcells

	### add relative position info
	for key in rel_pos[bundle_no].keys():
		sum_df.loc[iR, key] = rel_pos[bundle_no][key]

	### add grid angle value data
	sum_df = dataframe_add_column(sum_df, annot_angle_cols)
	sum_df.loc[iR, annot_angle_cols] = annots_df_current.loc[bundle_no, annot_angle_cols]

	### add QC columns
	sum_df.loc[iR, qc_cols] = annots_df_current.loc[bundle_no, qc_cols]
	sum_df.loc[iR, 'symmetry'] = symmetry

	### add positions of T3, T4, and T7 from heel.
	for col_type in [3,4,7]:
		cols = [f'T{col_type}l', f'T{col_type}c', f'T{col_type}h']
		new_cols = [f'{i}_fromheel' for i in cols]
		sum_df = dataframe_add_column(sum_df, new_cols)
		if(col_type == 3):
			sum_df.loc[iR, new_cols] = sum_df.loc[iR, cols].values - rel_pos[bundle_no]['R3']
		elif(col_type == 7):
			sum_df.loc[iR, new_cols] = sum_df.loc[iR, cols].values - rel_pos[bundle_no]['R4']
		elif(col_type == 4):
			if(r_type == 3):
				sum_df.loc[iR, new_cols] = sum_df.loc[iR, cols].values - rel_pos[bundle_no]['R3']
			elif(r_type == 4):
				sum_df.loc[iR, new_cols] = sum_df.loc[iR, cols].values - rel_pos[bundle_no]['R4']
	sum_df.loc[iR, 'T4l_fromheel']

	### get growth cone angle and length from tip to heel
	cat_angle = 'angle'
	cat_length = 'length'
	
	gc_length, gc_angle_rel = get_gc_angle_length(sum_df.loc[iR,:], coord_heels, phi_unit_real, cat_angle, cat_length, r_type)
	
	sum_df.loc[iR, f"{cat_length}_gc"] = gc_length

	sum_df.loc[iR, f"{cat_angle}_gc"] = gc_angle_rel
	sum_df.loc[iR, f"{cat_angle}_gc_plot"] = gc_angle_rel * phi_unit_theory
	
	if(r_type == 4):
		sum_df.loc[iR, f"{cat_angle}_gc_mrr"] = 0 - gc_angle_rel
	elif(r_type == 3):
		sum_df.loc[iR, f"{cat_angle}_gc_mrr"] = gc_angle_rel

	return sum_df


### processing data structure with annotated growth cone length and angle, and update bundle annotation data structure at the same time.
def process_sum_df(sum_df_old, annots_df, rel_poses, is_ml):
	"""
	Function: processing Dataframe with annotated growth cone length and angle, and update bundle annotation data structure at the same time.
	Inputs:
	- sum_df_old: DataFrame. Imported angle and length dataframe.
	- annots_df_old: DataFrame. Imported annotation csv dataframe.
	- rel_poses: Dictionary. Relative position info from the image quantification process.
	- is_ml: Boolean. whether or not to calculate repulsion model - related values.
	Output:
	- sum_df: DataFrame. processed DataFrame that contains both bundle heel and target info and growth cone length and angle info.
	"""

	### get phi_unit
	criteria = (annots_df['is_Edge'] == 0) & (annots_df['symmetry'] <= 0.5)
	phi_unit_avg = get_angle_unit_data(annots_df, criteria = criteria)
	phi_unit_theory = get_angle_unit_theory('aTiCT4')[2]
	# print(phi_unit_avg, phi_unit_theory)

	### new sum_df dataframe with added columns
	sum_df = sum_df_old.copy(deep = True)
	paths = settings.paths

	qc_cols = group_headers(annots_df, 'is_', True)
	cols_add = ['heel_pos_type', 'bundle_rcells_total', 'length_fromheel']
	cols_add += qc_cols
	sum_df = dataframe_add_column(sum_df, cols_add)

	### group by time and sample ID
	annots_df_group = annots_df.groupby(['time_id', 'sample_no'])
	sum_df_group = sum_df.groupby(['time_id', 'sample_no'])

	### process each sample
	for key in rel_poses.keys():
		time_id = key[0]
		sample_no = key[1]
		rel_pos = rel_poses[key]
		print(f"{time_id}, {sample_no}", end = "; ")

		# if((time_id, sample_no) not in sum_df_group.groups):
		# 	print(f"ERROR! {time_id}hrs_smp{sample_no} not in sum_df!")
		if((time_id, sample_no) in sum_df_group.groups):
			### sum_df
			sum_df_current = sum_df_group.get_group((time_id, sample_no))
			sum_df_current_gp = sum_df_current.groupby('bundle_no')

			### annots_df
			annots_df_current = annots_df_group.get_group((time_id, sample_no))
			annots_df_current.loc[:,'bundle_no'] = annots_df_current.loc[:,'bundle_no'].values.astype(int)
			annots_df_current = annots_df_current.reset_index().set_index('bundle_no')

			### process each bundle
			for bundle_no in annots_df_current.index:
				
				### bundle geometry information.
				phi_range_1 = rel_pos[bundle_no]["phi_range_1"]
				phi_range_2 = rel_pos[bundle_no]["phi_range_2"]
				symmetry = abs(phi_range_1 - phi_range_2)/max(phi_range_2, phi_range_1)

				### heel and target grid
				ch = get_heel_coords_sum(bundle_no, annots_df_current)
				ct = get_target_coords_sum(bundle_no, annots_df_current)

				### relative positions info
				if(bundle_no not in rel_pos.keys()):
					print(f"ERROR! Bundle No.{bundle_no} don't exist in output_data!")
				else:
					r3_heel = rel_pos[bundle_no]['R3']
					r4_heel = rel_pos[bundle_no]['R4']
					t3_pos = rel_pos[bundle_no]['T3c']
					t7_pos = rel_pos[bundle_no]['T7c']

				### matching summary_df with bundles_df
				inds_sum =  sum_df_current.index[(sum_df_current['bundle_no'] == bundle_no)]

				### Error: more than two R cells recorded for the particular bundle.
				if(len(inds_sum) > 2):
					print(f'Error! multiple incidents (n = {inds_sum}) of same bundle! bundle_no = {bundle_no}')

				### normal
				elif((len(inds_sum) > 0) & (len(inds_sum) <= 2)):

					r_types = sum_df_current.loc[inds_sum,['type_Rcell']]
					num_rcells = len(inds_sum)

					#### R3R4 case
					if(sum_df_current.loc[inds_sum,['type_bundle']].values.flatten()[0] == 'R3R4'): 
						for iR in r_types.index:
							r_type = r_types.loc[iR, 'type_Rcell']

							if(r_type == 3):
								sum_df.loc[iR, 'heel_pos_type'] = 3
								sum_df.loc[iR, 'length_fromheel'] = sum_df.loc[iR, 'length'] - r3_heel
							elif(r_type == 4):
								sum_df.loc[iR,'heel_pos_type'] = 4
								sum_df.loc[iR, 'length_fromheel'] = sum_df.loc[iR, 'length'] - r4_heel
							else:
								print('EROR! Not R3 nor R4!')
							
							if(sum_df.loc[iR, 'angle'] < 0):
								phi_unit_real = phi_range_1
							else:
								phi_unit_real = phi_range_2
							
							sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR, r_type, phi_unit_real, phi_unit_theory)
							
							#### mutual repulsion
							if(is_ml): 
								##### grid in standardized coordinates
								target_stc_car, heels_stc_car, _ = get_cartasian_grid_from_stc(sum_df.loc[iR,:], ch, 'angle', 'length', phi_unit_avg)
								# print(f"phi_unit_avg={phi_unit_avg}")
								# print(f"heels_stc_car={heels_stc_car}")

								##### get theoretical angles
								point, vector, theta, angle, vs = calculate_mutual_repulsion_theory(heels_stc_car, target_stc_car, r_type)
								# print(f"theta={theta}, angle={angle}.")

								sum_df.loc[iR, f'ml_theory_theta'] = theta
								sum_df.loc[iR, f'ml_theory_angle'] = angle
								sum_df.loc[iR, f'ml_theory_vec_x'] = vector[0]
								sum_df.loc[iR, f'ml_theory_vec_y'] = vector[1]
								for i in range(vs.shape[0]):
									sum_df.loc[iR, f'ml_x_v{i+1}'] = vs[i,0]
									sum_df.loc[iR, f'ml_y_v{i+1}'] = vs[i,1]
								
								#### get reference points
								if(r_type == 3):
									theta_ref = inner_angle(target_stc_car[2,:] - heels_stc_car[2,:], heels_stc_car[3,:] - heels_stc_car[2,:], True)
									angle_ref = inner_angle(target_stc_car[2,:] - heels_stc_car[2,:], np.array([1,0]), True)
								elif(r_type == 4):
									theta_ref = inner_angle(target_stc_car[5,:] - heels_stc_car[3,:], heels_stc_car[2,:] - heels_stc_car[3,:], True)
									angle_ref = inner_angle(target_stc_car[5,:] - heels_stc_car[3,:], np.array([1,0]), True)
								sum_df.loc[iR, 'theta_ref'] = theta_ref
								sum_df.loc[iR, 'angle_ref'] = angle_ref

								#### get measured angles
								cat_angle = 'angle'
								cat_length = 'length'
								gc_point, gc_vector, gc_theta, gc_angle = calculate_mutual_repulsion_data(sum_df.loc[iR,:], ch, phi_unit_avg, cat_angle, cat_length, r_type)

								sum_df.loc[iR, f'ml_actual_theta'] = gc_theta
								sum_df.loc[iR, f'ml_actual_angle'] = gc_angle

								sum_df.loc[iR, f'ml_x_vgc'] = gc_vector[0]
								sum_df.loc[iR, f'ml_y_vgc'] = gc_vector[1]

					#### R3/R3 or R4/R4 case:
					else:
						angle1 = sum_df.loc[r_types.index[0], 'angle']
						angle2 = sum_df.loc[r_types.index[1], 'angle']
						# print(angle1, angle2, iR3, iR4, end = "; ")
						if(angle1 > angle2):
							iR3 = r_types.index[0]
							iR4 = r_types.index[1]
						else:
							iR3 = r_types.index[1]
							iR4 = r_types.index[0]
						sum_df.loc[iR3,'heel_pos_type'] = 3
						sum_df.loc[iR4,'heel_pos_type'] = 4

						sum_df.loc[iR3, 'length_fromheel'] = sum_df.loc[iR3, 'length'] - r3_heel
						sum_df.loc[iR4, 'length_fromheel'] = sum_df.loc[iR4, 'length'] - r4_heel
						
						sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR3, 3, phi_range_2, phi_unit_theory)
						sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR4, 4, phi_range_1, phi_unit_theory)
			
		sum_df_groups = sum_df.groupby(['heel_pos_type', 'type_bundle'])
		if((3, 'R3R3') in sum_df_groups.groups.keys()):
			sum_df.loc[sum_df_groups.get_group((3, 'R3R3')).index,'type_plot'] = 'R3/R3(3)'
		if((4, 'R3R3') in sum_df_groups.groups.keys()):
			sum_df.loc[sum_df_groups.get_group((4, 'R3R3')).index,'type_plot'] = 'R3/R3(4)'
		if((3, 'R4R4') in sum_df_groups.groups.keys()):
			sum_df.loc[sum_df_groups.get_group((3, 'R4R4')).index,'type_plot'] = 'R4/R4(3)'
		if((4, 'R4R4') in sum_df_groups.groups.keys()):
			sum_df.loc[sum_df_groups.get_group((4, 'R4R4')).index,'type_plot'] = 'R4/R4(4)'
		sum_df.loc[sum_df_groups.get_group((3, 'R3R4')).index,'type_plot'] = 'R3'
		sum_df.loc[sum_df_groups.get_group((4, 'R3R4')).index,'type_plot'] = 'R4'

	return sum_df


# ================= Figure 3B =================
### stats of Figure 3B.
def stats_fig3b(df, hue_name, value_name, pair_list, method):
	"""
	Function: get sample size, test for normality, and test for difference between R3 and R4 relative angle measurements.
	
	Inputs:
	- df: dataframe containing grouping and data. 
	- hue_name: column name of grouping.
	- value_name: column name of data.
	- pair_list: groups (R3 vs. R4).
	- method: test method to use. 'Mann-Whitney'/'Student T'/'Welch'/'KS'
	
	Output:print out sample size and stats.
	"""
	df_groups = df.groupby(hue_name)
	for pair in pair_list:
		a = df_groups.get_group(pair[0]).loc[:,value_name].values
		b = df_groups.get_group(pair[1]).loc[:,value_name].values
		t, pa = ss.kstest(a, 'norm')
		t, pb = ss.kstest(b, 'norm')
		if(method == 'Mann-Whitney'):
			t, p = ss.mannwhitneyu(a, b, alternative = 'two-sided')
		elif(method == 'Student T'):
			t, p = ss.ttest_ind(a,b)
		elif(method == 'Welch'):
			t, p = ss.ttest_ind(a,b, equal_var = False)
		elif(method == 'KS'):
			t, p = ss.ks_2samp(a, b, alternative='two-sided', mode='auto')
		print(f'count: {pair[0]} = {len(a)}, num {pair[1]} = {len(b)}')
		print(f'KS normality test: pa = {pa}, pb = {pb}')
		print(f'{method} test: {p}')

# ================= Figure 4 =================
### Plotting
def generate_summary_polar_figure(plot_df, pert_info, **kwargs):
	"""
	Function: plot polar line plots for relative length and angles for sev>Fz and sev>Nic flies.

	Inputs:
	- plot_df: dataframe containing relative lengths and angles data for a specific perturbation group (sev>Fz 24hrs, sev>Fz 28hrs, sev>Nic 24hrs, sev>Nic 28 hrs)
	- pert_info: information about the perturbation group
		- time_id: age.
		- pert_cat: genetics. "Fz"/"Nic".
		- pert_type: which type of perturbed bundle occurs. "R3/R3" or "R4/R4"
		- pert_rtype. which R-cell-type exist in perturbed bundles. "R3" or "R4"
	- additional inputs:
		- is_save: Boolean. Save figures or not. Default = False.
		- fig_format: extension figure format. Default = "svg".
		- fig_res: figure resolution. Default = 300.

	Output: Figure.
	"""

	### unravel params
	time_id, pert_cat, pert_type, pert_rtype = pert_info
	paths = settings.paths
	matching_info = settings.matching_info
	color_code = matching_info.color_code
	
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('fig_format' in kwargs.keys()):
		fig_format = kwargs['fig_format']
	else:
		fig_format = 'svg'
	if('fig_res' in kwargs.keys()):
		fig_res = kwargs['fig_res']
	else:
		fig_res = 300

	
	### get data
	theta_cat = 'angle_gc_plot'
	r_cat = 'length_gc'
	
	sum_df_grouped = plot_df.groupby('type_plot')
	
	ind_r3 = sum_df_grouped.get_group('R3').index
	ind_r4 = sum_df_grouped.get_group('R4').index
	if(pert_cat == 'Nic'):
		ind_pert_r3 = sum_df_grouped.get_group('R4/R4(3)').index
		ind_pert_r4 = sum_df_grouped.get_group('R4/R4(4)').index
	elif(pert_cat == 'Fz'):
		ind_pert_r3 = sum_df_grouped.get_group('R3/R3(3)').index
		ind_pert_r4 = sum_df_grouped.get_group('R3/R3(4)').index

	pos_t3 = np.mean(plot_df.loc[:,'T3c'].values - plot_df.loc[:,'R3'].values)
	pos_t7 = np.mean(plot_df.loc[:,'T7c'].values - plot_df.loc[:,'R4'].values)
	pos_t4 = np.mean(plot_df.loc[:,'T4c'].values - plot_df.loc[:,'R4'].values)

	phi_unit = get_angle_unit_theory('phi_unit')

	thetas = {
		'R3':plot_df.loc[ind_r3,theta_cat].values,
		'R4':plot_df.loc[ind_r4,theta_cat].values,
		'pert_R3':plot_df.loc[ind_pert_r3,theta_cat].values,
		'pert_R4':plot_df.loc[ind_pert_r4,theta_cat].values,
	}
	rs = {
		'R3':plot_df.loc[ind_r3,r_cat].values,
		'R4':plot_df.loc[ind_r4,r_cat].values,
		'pert_R3':plot_df.loc[ind_pert_r3,r_cat].values,
		'pert_R4':plot_df.loc[ind_pert_r4,r_cat].values,
	}

	dTiCs = {3:pos_t3, 7:pos_t7, 4: pos_t4}
	target_grid_polar = get_target_grid_polar_summary(return_type = 'theory', dTiCs = dTiCs)
	
   
	
	### figure set-up
	legend = ['R3', 'R4', pert_type]
	plot_line = {
		'R3':'-',
		'R4':'-',
		'pert_R3':'--',
		'pert_R4':'--',
	}
	plot_color = {
		'R3':color_code[3],
		'R4':color_code[4],
		'pert_R3':color_code[pert_rtype],
		'pert_R4':color_code[pert_rtype],
	}
	
	SMALL_SIZE = 20
	MEDIUM_SIZE = 24
	BIGGER_SIZE = 28
	plt.rc('font', size=SMALL_SIZE)         # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)   # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)   # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title

	fig_name = f'Figure4_{pert_cat}_{time_id}hrs_polar.{fig_format}'
	fig_save_path = os.path.join(paths.output_prefix, fig_name)
	
	### plotting
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_axes([0.1, 0.1, 0.75, 0.79], polar=True)

	# plot data
	for i in thetas.keys():
		ax.errorbar(np.mean(thetas[i]), np.mean(rs[i]), xerr = np.std(thetas[i]), 
					yerr = np.std(rs[i]), color = 'k', elinewidth=1)
		ax.plot([0,np.mean(thetas[i])], [0,np.mean(rs[i])], linewidth = 1.5, 
				linestyle = plot_line[i], color = plot_color[i])


	# plot targets
	for i in [2,3,5]:
		ax.plot(target_grid_polar[i,0], target_grid_polar[i,1], 'o', 
				color = color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 1.5, mfc = 'none')


	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 8, box.height])

	# axis settings
	ax.set_thetamin(-35)
	ax.set_thetamax(35)
	ax.set_rlim(0, 1.8)
	ax.tick_params(axis = 'x', labelbottom = True)
	ax.tick_params(axis = 'y', labelleft = True)

	plt.yticks(ticks = [0, 0.4, 0.8, 1.2, 1.6])
	plt.xticks(ticks = [-phi_unit, 0, phi_unit], labels = [-1, 0, 1])
	ax.set_xlabel("Relative Length (a.u.)")
	ax.yaxis.set_label_position("right")
	plt.ylabel("Relative Angle (a.u.)", labelpad=35)
	plt.title(f"sev>{pert_cat}, {time_id} hrs")


	ax.grid(axis = 'y', linestyle = '--', which = 'major', linewidth=0.8)
	ax.grid(axis = 'x', linestyle = '--', which = 'major', linewidth=0.8)
	ax.grid(True)

	if(is_save):
		plt.savefig(fig_save_path, dpi=300, bbox_inches='tight', format = fig_format)
	plt.show()


### stats.
def stats_fig4(df_current, x_cat, pert_cat, time_id, which_ycat):
	"""
	Function: get sample size and p-values for relative length and angle data for sev>Fz and sev>Nic flies.

	Inputs:
	- df_current: dataframe containing grouping and data.
	- x_cat: column name of grouping.
	- pert_cat: genetics. "Fz"/"Nic".
	- time_id: age. 24/28.
	- which_ycat: "count"/"angle"/"length", indicate whether to print sample size or do comparison on relative angle/length data.

	Output: print out sample size and stats.
	"""

	### initialization
	if(pert_cat == 'Fz'):
		pert_type = 'R3/R3'
	elif(pert_cat == 'Nic'):
		pert_type = 'R4/R4'

	if(which_ycat == 'count'):
		df_count = df_current.groupby(x_cat).count()
		y_cat = 'bundle_no'
	elif(which_ycat == 'angle'):
		y_cat = 'angle_gc_mrr'
		inds = ['R3', f'{pert_type}(3)', 'R3', 'R4']
		cols = ['R4', f'{pert_type}(4)', f'{pert_type}(3)', f'{pert_type}(4)']
	elif(which_ycat == 'length'):
		y_cat = 'length_gc'
		if(pert_cat == 'Fz'):
			inds = ['R3', f'{pert_type}(3)', 'R3', 'R3']
		elif(pert_cat == 'Nic'):
			inds = ['R3', f'{pert_type}(3)', 'R4', 'R4']
		cols = ['R4', f'{pert_type}(4)', f'{pert_type}(3)', f'{pert_type}(4)']
	
	### calculate p-values
	data = [df_current.loc[ids, y_cat].values for ids in df_current.groupby(x_cat).groups.values()]
	H, p = ss.kruskal(*data)
	df_stat = sp.posthoc_mannwhitney(df_current, val_col=y_cat, group_col=x_cat, p_adjust = 'holm')

	### printing.
	if(which_ycat == 'count'):
		print(f"==={pert_cat}_{time_id}hrs_count===")
		for i in range(len(df_count)):
			print(f"{df_count.index[i]}: {df_count.iloc[i]['bundle_no']}")
	elif(which_ycat == 'length'):
		print(f"==={pert_cat}_{time_id}hrs_length===")
		for i in range(len(inds)):
			print(f"{inds[i]} vs {cols[i]}: {df_stat.loc[inds[i], cols[i]]}")
	elif(which_ycat == 'angle'):
		print(f"==={pert_cat}_{time_id}hrs_angle===")
		for i in range(len(inds)):
			print(f"{inds[i]} vs {cols[i]}: {df_stat.loc[inds[i], cols[i]]}")

# ================= Polar Density Plot: Fig S2 & S5 =================
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
	
	phi_range = inner_angle(slice_one_point - center_point, slice_zero_point - center_point, True)
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

	phi_start = angle_normalization(phi_start)
	phi_end = angle_normalization(phi_end)

	phis = get_phis(phi_start, phi_end, num_angle_section + num_outside_angle*2 + 2)

	if(smallest_angle(angle_start_to_r, phis[-1]) < smallest_angle(angle_start_to_r, phis[0])):
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
	c = line_intersection((r_heels_cart[2,:], target_grid_cart[2,:]),(r_heels_cart[3,:], target_grid_cart[5,:]))

	#### after standardization
	dTiC = np.zeros((6,1))
	for i in range(1,6):
		dTiC[i] = np.linalg.norm(target_grid_cart[i,:] - c)
	dTiC = dTiC/dTiC[3]
	aTiCT4 = np.zeros((6,1))
	for i in range(1,6):
		aTiCT4[i] = inner_angle(target_grid_cart[i,:] - c, target_grid_cart[3,:] - c, True)
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

### polar density plot
def plot_polar_density(matrix, channel_no, rel_points, analysis_params, **kwargs):
	### unravel params
	matching_info = settings.matching_info

	if('plot_type' in kwargs.keys()):
		plot_type = kwargs['plot_type']
	else:
		plot_type = 'max'
	if('figsize' in kwargs.keys()):
		figsize = kwargs['figsize']
	else:
		figsize = (10,8)
	if('theta_lim' in kwargs.keys()):
		theta_lim = kwargs['theta_lim']
	else:
		theta_lim = [-30, 30]
	if('rlim' in kwargs.keys()):
		rlim = kwargs['rlim']
	else:
		rlim = [0, 2.5]
	if('cbar_axis' in kwargs.keys()):
		cbar_axis = kwargs['cbar_axis']
	else:
		cbar_axis = [0.8, 0.2, 0.02, 0.6]
	
	### get matrix
	if(plot_type == 'max'):
		mm = np.max(matrix[channel_no,:,:,:], axis = 2)
		
	elif(plot_type == 'mean'):
		mm = np.mean(matrix[channel_no,:,:,:], axis = 2)
	print(mm.shape)

	### get polar plot values
	colormap = plt.get_cmap('gray')
	colormap.set_bad('snow')

	thetav, rv, z1, norm1, target_grid_polar = get_polar_plot_values(analysis_params, channel_no, mm, colormap, rel_points)
	
	mask = z1 == 0
	zm = ma.masked_array(z1, mask=mask)
	if('vmax' in kwargs.keys()):
		vmax = kwargs['vmax']
	else:
		vmax = np.percentile(zm, 99)

	phi_unit = get_angle_unit_theory('phi_unit')
	

	### figure
	sns.set_style('ticks')
	fig = plt.figure(figsize = figsize)
	ax2 = fig.add_subplot(111, polar = True)

	## plot value
	sc = ax2.pcolormesh(thetav, rv, zm, cmap=colormap, vmin = 0, vmax = vmax)

	## plot angle reference
	ax2.plot([0, target_grid_polar['c'][matching_info.target_id_to_index[3],0]], 
			 [0, rlim[1]], 
			 '--', color = 'lightgray', linewidth = 1)
	ax2.plot([0, target_grid_polar['c'][matching_info.target_id_to_index[4],0]], 
			 [0, rlim[1]], 
			 '--', color = 'lightgray', linewidth = 1)
	ax2.plot([0, target_grid_polar['c'][matching_info.target_id_to_index[7],0]], 
			 [0, rlim[1]], 
			 '--', color = 'lightgray', linewidth = 1)
	
	## plot target position
	for i in [0,2,3,5]:
		ax2.plot(target_grid_polar['c'][i,0], target_grid_polar['c'][i,1], 'o', 
				 color = matching_info.color_code[matching_info.index_to_target_id[i]], 
				 markersize = 30, mew = 2, mfc = 'none')


	## set polar to pie
	ax2.set_thetamin(theta_lim[0])
	ax2.set_thetamax(theta_lim[1])
	ax2.set_rlim(rlim)
	ax2.tick_params(axis = 'y', labelsize = 30, pad = -4)
	ax2.tick_params(axis = 'x', labelsize = 30, pad = 6)

	if('r_tick' in kwargs.keys()):
		ax2.set_yticks(kwargs['r_tick'])
	if('theta_tick' in kwargs.keys()):
		ax2.set_xticks(kwargs['theta_tick'])
	else:
		ax2.set_xticks(ticks = [-phi_unit, 0, phi_unit])
		ax2.set_xticklabels(labels = [-1, 0, 1])
	if(channel_no == 0):
		ax2.set_title('R3 or R4', fontsize = 30)
	elif(channel_no == 1):
		ax2.set_title('R4', fontsize = 30)

	#### color bar for polar plot
	cNorm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)   #-- Defining a normalised scale
	ax5 = fig.add_axes(cbar_axis)       #-- Creating a new axes at the right side
	if('cbar_tick' in kwargs.keys()):
		cb1 = matplotlib.colorbar.ColorbarBase(ax5, norm=cNorm, cmap=colormap, ticks=kwargs['cbar_tick'])    #-- Plotting the colormap in the created axes
	else:
		cb1 = matplotlib.colorbar.ColorbarBase(ax5, norm=cNorm, cmap=colormap) #-- Plotting the colormap in the created axes
	cb1.ax.tick_params(labelsize=30)
	fig.subplots_adjust(left=0.0,right=0.95)
	
	return fig

### generate figure
def generate_density_plot_figure(output_data, **kwargs):
	"""
	Function: plot density plots of a specific bundle.

	Inputs:
	- output_data: Dictionaries with intensity matrix and parameters of representative bundles.
		- keys = 'figure_ID', 'time_ID', 'bundle_type', 'category_ID', 'parameter', 'relative_positions', 'intensity_matrix', 'analysis_params_general'
		- 'figure_ID', 'time_ID', 'category_ID', 'bundle_type': general information about this particular bundle.
		- 'relative_positions': dictionary. relative positions of targets and heels of this particular bundle.
		- 'intensity_matrix': numpy array. GFP and RFP density map of this particular bundle.
		- 'analysis_params_general': class of settings.GeneralParams. parameters used when doing image quantification
	- Additional inputs:
		- is_save: Boolean. Save figures or not. Default = False.
		- channels: Which channel (or channels) to plot. Default = ["GFP", "RFP"]
		- fig_format: extension figure format. Default = "svg".
		- fig_res: figure resolution. Default = 300.
		- fig_name: name of the figure.

	Output: Figure.
	"""

	### unravel params
	paths = settings.paths

	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('fig_format' in kwargs.keys()):
		fig_format = kwargs['fig_format']
	else:
		fig_format = 'svg'
	if('fig_res' in kwargs.keys()):
		fig_res = kwargs['fig_res']
	else:
		fig_res = 300
	if('channels' in kwargs.keys()):
		channels = kwargs['channels']
	else:
		channels = ['GFP', 'RFP']
	
	
	rel_points = output_data['relative_positions']
	matrix = output_data['intensity_matrix']
	category = output_data['category_ID']
	fig_id = output_data['figure_ID']
	time_id = output_data['time_ID']
	bundle_type = output_data['bundle_type']
	

	
	num_angle_section = output_data['analysis_params_general'].num_angle_section
	num_outside_angle = output_data['analysis_params_general'].num_outside_angle
	radius_expanse_ratio = output_data['analysis_params_general'].radius_expanse_ratio[1]
	num_x_section = output_data['analysis_params_general'].num_x_section
	z_offset = output_data['analysis_params_general'].z_offset

	analysis_params = (num_angle_section, 
					   num_outside_angle, 
					   num_x_section,
					   z_offset,
					   radius_expanse_ratio)
	
	plot_type = 'mean'
	figsize = (8,8)
	cbar_axis = [0.85, 0.2, 0.03, 0.6]
	r_tick = [0,1,2]
	theta_lim = [-40, 40]
	rlim = [0, 2.2]
	

	### figures
	for channel in channels:
		
		if(channel == 'RFP'):
			channel_no = 0
			if('rfp_cmap' in kwargs.keys()):
				[vmax, cbar_tick] = kwargs['rfp_cmap']
			else:
				vmax = np.nan
		elif(channel == 'GFP'):
			channel_no = 1
			if('gfp_cmap' in kwargs.keys()):
				[vmax, cbar_tick] = kwargs['gfp_cmap']
			else:
				vmax = np.nan

		if('fig_name' in kwargs.keys()):
			name = kwargs['fig_name']
			fig_name = f'{name}_{channel}_density_plot.{fig_format}'
		else:
			fig_name = f'{fig_id}_{category}_{time_id}hrs_{bundle_type}_{channel}_density_plot.{fig_format}'

		if not (np.isnan(vmax)):
			fig = plot_polar_density(matrix, channel_no, rel_points, 
							 plot_type = plot_type, figsize = figsize, analysis_params = analysis_params, 
							 r_tick = r_tick, theta_lim = theta_lim, rlim = rlim, 
							 cbar_axis = cbar_axis, vmax = vmax, cbar_tick = cbar_tick)
		else:
			fig = plot_polar_density(matrix, channel_no, rel_points, 
							 plot_type = plot_type, figsize = figsize, analysis_params = analysis_params, 
							 r_tick = r_tick, theta_lim = theta_lim, rlim = rlim, 
							 cbar_axis = cbar_axis)
		
		if(is_save):
			fig_save_path = os.path.join(paths.output_prefix, fig_name)
			plt.savefig(fig_save_path, dpi = fig_res, bbox_inches = 'tight', format = fig_format)
		plt.show()

# ================= S3 Fig ================= #
### S3B Fig. 
def plot_sample_variation_raw_measurements(annots_df_group, **kwargs):
	"""
	Function: plot raw coordinate measurements of heel and target positions of wild-type flies of a specific age. bundles from one sample are plotted together on the same subplot.

	Inputs:
	- annots_df_group: DataFrame group. Processed annotation information of a specific age, grouped by sample number.
	- Additional inputs:
		- is_save: Boolean. Save figures or not. Default = False.
		- fig_format: extension figure format. Default = "svg".
		- fig_res: figure resolution. Default = 300.

	Output: Figure.
	"""


	### parameters
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('fig_format' in kwargs.keys()):
		fig_format = kwargs['fig_format']
	else:
		fig_format = 'svg'
	if('fig_res' in kwargs.keys()):
		fig_res = kwargs['fig_res']
	else:
		fig_res = 300

	paths = settings.paths
	index_to_target_id = settings.matching_info.index_to_target_id
	color_code =  settings.matching_info.color_code


	### create figure.
	num_subplots = len(annots_df_group)
	fig, axes = plt.subplots(num_subplots, 1, figsize = (50, 10))
	fig.tight_layout()

	heel_coords = {}
	target_coords = {}

	### loop through samples.
	for i_fig in range(num_subplots):
		i_sample = i_fig
		print(f"Sample No.{i_sample+1}: ", end = "")
		
		### calculating
		sample_id = list(annots_df_group.groups.keys())[i_sample]
		annots_df_current = annots_df_group.get_group(sample_id).reset_index(drop = True)
		annots_df_current.set_index('bundle_no', drop = True, inplace = True)
		
		heel_coords[i_fig] = np.zeros((6,2, len(annots_df_current)))
		target_coords[i_fig] = np.zeros((6,2, len(annots_df_current)))

		
		for ind, bundle_no in enumerate(annots_df_current.index):
			print(f"{bundle_no}-", end = "")
			ch = get_heel_coords_sum(bundle_no, annots_df_current)
			ct = get_target_coords_sum(bundle_no, annots_df_current, is_converted = True)
			heel_coords[i_fig][:,:,ind] = ch
			target_coords[i_fig][:,:,ind] = ct
					
		### Plotting
		ax = axes.ravel()[i_fig]

		#### plot x axis 
		ax.plot([-5, 20], [0,0], '--', color = 'gray')
		
		#### plot dots
		for i in range(heel_coords[i_fig].shape[2]):
			for j in range(6):
				#### R cells
				ax.plot(heel_coords[i_fig][j,0,i], heel_coords[i_fig][j,1,i], 
						'o', color = color_code[j+1], markersize = 5, alpha = 0.5)
				
				#### L cells
				ax.plot(target_coords[i_fig][j,0,i], target_coords[i_fig][j,1,i], 
						'o', mec = color_code[index_to_target_id[j]], markersize = 15, mew = 1, alpha = 0.8, mfc = 'none')
			ax.plot(0,0,'o', color = 'gray', markersize = 10)

		#### plot center.
		ax.plot(0, 0, 'o', color = 'k', markersize = 5)
		ax.text(0.3, -1, "C")
		
		#### axis
		ax.set_xlim([-5, 20])
		ax.set_ylim([-6, 6])
		ax.set_aspect(aspect=1)
		ax.set_yticks([-5, 0, 5])

		if(i_fig != num_subplots-1): ### last sub-figure
			ax.tick_params(axis = 'x', labelbottom = [])
		else:
			ax.tick_params(axis = 'x', labelbottom = [-5, 0, 5, 10, 15, 20])
			ax.set_xlabel('X (um)')

		if(i_fig == round(num_subplots/2)-1): ### middle sub-figure.
			ax.set_ylabel(f"Sample No. {i_sample+1}\nY (um)")
		else:
			ax.set_ylabel(f"Sample No. {i_sample+1}\n")


		print("")
	
	### saving
	fig_name = f'S3B_Fig.{fig_format}'
	fig_save_path = os.path.join(paths.output_prefix, fig_name)
	if(is_save):
		plt.savefig(fig_save_path, dpi=fig_res, bbox_inches='tight', format = fig_format, transparent=False)
	plt.show()

	return heel_coords, target_coords

### S3C Fig.
def plot_sample_variation_polar(annots_df_group, **kwargs):
	"""
	Function: plot polar coordinate values of R3, R4, T3, T4, T3' positions of wild-type flies of a specific age. bundles from one sample are plotted together on the same subplot.

	Inputs:
	- annots_df_group: DataFrame group. Processed annotation information of a specific age, grouped by sample number.
	- Additional inputs:
		- is_save: Boolean. Save figures or not. Default = False.
		- fig_format: extension figure format. Default = "svg".
		- fig_res: figure resolution. Default = 300.

	Output: 
	- Figure.
	- sum_coords: summary of polar coordinates.
	"""


	### parameters
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('fig_format' in kwargs.keys()):
		fig_format = kwargs['fig_format']
	else:
		fig_format = 'svg'
	if('fig_res' in kwargs.keys()):
		fig_res = kwargs['fig_res']
	else:
		fig_res = 300

	### Params
	paths = settings.paths
	phi_unit = get_angle_unit_theory('phi_unit')
	color_code =  settings.matching_info.color_code
	plot_color = {
		'R3':color_code[3],
		'R4':color_code[4],
		'T4':color_code[4],
		'T3':color_code[3],
		'T7':color_code[7],
	}
	num_subplots = len(annots_df_group)

	### Figure set-up
	fig, axes = plt.subplots(num_subplots, 1, figsize = (30, 15), subplot_kw={'projection': 'polar'})
	fig.tight_layout()

	sum_coords = {}
	coords = {}
	for i in plot_color.keys():
		sum_coords[i] = np.zeros((2, num_subplots))

	for i_fig in range(num_subplots):
		i_sample = i_fig
		
		### calculating
		sample_id = list(annots_df_group.groups.keys())[i_sample]
		annots_df_current = annots_df_group.get_group(sample_id).reset_index(drop = True)
		annots_df_current.set_index('bundle_no', drop = True, inplace = True)
		
		### initialization
		coords[i_fig] = {}
		for i in plot_color.keys():
			coords[i_fig][i] = np.zeros((2, len(annots_df_current)))
		
		### loop through bundle
		for ind, bundle_no in enumerate(annots_df_current.index):
			pos_t3 = annots_df_current.loc[bundle_no, 'T3c']
			pos_t4 = 1
			pos_t7 = annots_df_current.loc[bundle_no, 'T7c']
			dTiCs = {3:pos_t3, 7:pos_t7, 4: pos_t4}
			target_grid_polar = get_target_grid_polar_summary(return_type = 'theory', dTiCs = dTiCs)
			coords[i_fig]['R3'][0, ind] = target_grid_polar[2,0]
			coords[i_fig]['R3'][1, ind] = annots_df_current.loc[bundle_no, 'R3']
			coords[i_fig]['R4'][0, ind] = target_grid_polar[5,0]
			coords[i_fig]['R4'][1, ind] = annots_df_current.loc[bundle_no, 'R4']
			coords[i_fig]['T3'][0, ind] = target_grid_polar[2,0]
			coords[i_fig]['T3'][1, ind] = annots_df_current.loc[bundle_no, 'T3c']
			coords[i_fig]['T7'][0, ind] = target_grid_polar[5,0]
			coords[i_fig]['T7'][1, ind] = annots_df_current.loc[bundle_no, 'T7c']
			coords[i_fig]['T4'][0, ind] = 0
			coords[i_fig]['T4'][1, ind] = 1
			
		### get centroids
		for t in coords[i_fig].keys():
			sum_coords[t][:, i_sample] = np.mean(coords[i_fig][t], axis = 1)
		
		### Plotting
		ax = axes.ravel()[i_fig]
		
		### references
		ax.plot([0,0], [0,2.5], '--', color = "0.8", linewidth = 0.5)
		ax.plot([0,target_grid_polar[2,0]], [0,2.5], '--', color = "0.8", linewidth = 0.5)
		ax.plot([0,target_grid_polar[5,0]], [0,2.5], '--', color = "0.8", linewidth = 0.5)

		### individual dots
		for ind in range(len(annots_df_current)):
			for t in ['R3', 'R4']:
				ax.plot(coords[i_fig][t][0, ind], coords[i_fig][t][1, ind], 
					 'o', color = plot_color[t], markersize = 10, alpha = 0.5)
			for t in ['T3', 'T4', 'T7']:
				ax.plot(coords[i_fig][t][0, ind], coords[i_fig][t][1, ind], 
					 'o', mec = plot_color[t], markersize = 25, mew = 1.0, mfc = 'none', alpha = 0.8)
		ax.plot(0, 0, 'o', color = 'k', markersize = 5)
		ax.text(0.3, -1, "C")
		
		### axis
		ax.set_thetamin(-30)
		ax.set_thetamax(30)
		ax.set_rlim(0, 2.5)
		ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
		ax.set_xticks([-phi_unit, 0, phi_unit])
		ax.set_xticklabels([1, 0, -1])
		ax.grid(axis = 'y', linestyle = '--', which = 'major', linewidth = 0.5)
		ax.grid(axis = 'x', linestyle = '--', which = 'major', linewidth = 0.5)
		
		ax.tick_params()
		
		if(i_fig == num_subplots-1): ### last sub-figure
			ax.set_xlabel('Relative Length (a.u.)')
		if(i_fig == round(num_subplots/2)-1): ### middle sub-figure.
			ax.set_ylabel("\nRelative Angle (a.u.)")
			ax.yaxis.set_label_position("right")

	### saving
	fig_format = 'svg'
	fig_name = f'S3C_Fig.{fig_format}'
	fig_save_path = os.path.join(paths.output_prefix, fig_name)
	if(is_save):
		plt.savefig(fig_save_path, dpi=fig_res, bbox_inches='tight', format = fig_format)
	plt.show()

	return coords, sum_coords

### S3D Fig
def plot_sample_variation_polar_centroids(sum_coords, **kwargs):
	"""
	Function: plot polar coordinate values of R3, R4, T3, T4, T3' positions of wild-type flies of a specific age. bundles from one sample are plotted together on the same subplot.

	Inputs:
	- sum_coords: summary of polar coordinates.
	- Additional inputs:
		- is_save: Boolean. Save figures or not. Default = False.
		- fig_format: extension figure format. Default = "svg".
		- fig_res: figure resolution. Default = 300.

	Output: Figure.
	"""

	### params
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('fig_format' in kwargs.keys()):
		fig_format = kwargs['fig_format']
	else:
		fig_format = 'svg'
	if('fig_res' in kwargs.keys()):
		fig_res = kwargs['fig_res']
	else:
		fig_res = 300


	num_subplots = sum_coords['R3'].shape[1]
	paths = settings.paths
	color_code =  settings.matching_info.color_code
	phi_unit = get_angle_unit_theory('phi_unit')
	plot_color = {
		'R3':color_code[3],
		'R4':color_code[4],
		'T4':color_code[4],
		'T3':color_code[3],
		'T7':color_code[7],
	}

	### set-up figure
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_axes([0.1, 0.1, 0.75, 0.79], polar=True)

	### plot references
	ax.plot([0, 0], [0,2.5], '--', color = "0.8", linewidth = 0.5)
	ax.plot([0, sum_coords["T3"][0,0]], [0,2.5], '--', color = "0.8", linewidth = 0.5)
	ax.plot([0, sum_coords["T7"][0,0]], [0,2.5], '--', color = "0.8", linewidth = 0.5)


	### plot summary dots
	for i_smp in range(3):
		for t in ['R3', 'R4']:
			#### dot
			ax.plot(sum_coords[t][0, i_smp], sum_coords[t][1, i_smp], 
				 'o', color = plot_color[t], markersize = 10, alpha = 0.5)

			#### text
			x = sum_coords[t][0, i_smp]
			y = sum_coords[t][1, i_smp]
			if(i_smp == 0):
				y -= 0.05
			elif(i_smp == 1):
				y += 0.05
			if(t == 'R3'):
				x *= 1.5
			ax.text(x, y, i_smp + 1, fontsize = 15)
			
		for t in ['T3', 'T4', 'T7']:
			#### dot
			ax.plot(sum_coords[t][0, i_smp], sum_coords[t][1, i_smp], 
				 'o', mec = plot_color[t], markersize = 25, mew = 1.0, mfc = 'none', alpha = 0.8)

			#### text
			if(t != 'T4'):
				if((t == 'T3') & (i_smp == 2)):
					ax.text(sum_coords[t][0, i_smp]*1.4, sum_coords[t][1, i_smp]+0.1, i_smp+1, fontsize=15)
				else:
					ax.text(sum_coords[t][0, i_smp]*1.4, sum_coords[t][1, i_smp], i_smp+1,fontsize=15)
		
	#### Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 8, box.height])

	#### axis
	ax.set_thetamin(-30)
	ax.set_thetamax(30)
	ax.set_rlim(0, 2.5)
	ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
	ax.set_xticks([-phi_unit, 0, phi_unit])
	ax.set_xticklabels([1, 0, -1])
	ax.grid(axis = 'y', linestyle = '--', which = 'major', linewidth = 0.5)
	ax.grid(axis = 'x', linestyle = '--', which = 'major', linewidth = 0.5)

	ax.set_xlabel("Relative Length (a.u.)")
	ax.yaxis.set_label_position("right")
	plt.ylabel("Relative Angle (a.u.)", labelpad=35)

	#### saving
	fig_name = f'S3D_Fig.{fig_format}'
	fig_save_path = os.path.join(paths.output_prefix, fig_name)
	if(is_save):
		plt.savefig(fig_save_path, dpi=fig_res, bbox_inches='tight', format = fig_format)
	
	plt.show()

# ================= S4 Fig ================= #
### S4A Fig
def plot_time_variation_raw_centroids(annots_df, **kwargs):
	"""
	Function: plot centroids of heel/target coordinates from all bundles of a given time point.

	Inputs:
	- annots_df: Dataframe. Processed annotations files.
	- Additional inputs:
		- is_save: Boolean. Save figures or not. Default = False.
		- fig_format: extension figure format. Default = "svg".
		- fig_res: figure resolution. Default = 300.

	Output:
	- Figure
	- heel_centroid_sum, target_centroid_sum: raw coordinate centroids of each time point.
	- coords_sum: standardized coordinates of each time point.
	"""

		### params
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('fig_format' in kwargs.keys()):
		fig_format = kwargs['fig_format']
	else:
		fig_format = 'svg'
	if('fig_res' in kwargs.keys()):
		fig_res = kwargs['fig_res']
	else:
		fig_res = 300

	paths = settings.paths
	color_code =  settings.matching_info.color_code
	index_to_target_id = settings.matching_info.index_to_target_id
	plot_color = {
		'R3':color_code[3],
		'R4':color_code[4],
		'T4':color_code[4],
		'T3':color_code[3],
		'T7':color_code[7],
	}

	### group by age
	annots_df_time_gp = annots_df.groupby(['time_id'])

	### initialization
	#### raw centroids
	heel_centroid_sum = np.zeros((6,2,len(annots_df_time_gp)))
	target_centroid_sum = np.zeros((6,2,len(annots_df_time_gp)))
	#### standardized centroids
	coords_sum = {}

	### figure set-up
	sns.set_style("whitegrid", {'axes.grid' : False})
	fig, axes = plt.subplots(4,2, figsize = (15, 20))

	### loop through time points
	for i_fig, time_id in enumerate(annots_df_time_gp.groups.keys()):
		print(f'{time_id} hrs: ', end = "")
		
		annots_df_time = annots_df_time_gp.get_group(time_id)
		annots_df_smp_gp = annots_df_time.groupby('sample_no')
		
		### initialize for coordinates of each time point
		heel_coords = np.zeros((6,2, len(annots_df_time)))
		target_coords = np.zeros((6,2, len(annots_df_time)))
		coords_sum[time_id] = {}
		for t in plot_color.keys():
			coords_sum[time_id][t] = np.zeros((2, len(annots_df_time)))
		
		### loop through sample
		ind = 0
		for sample_id in annots_df_smp_gp.groups.keys():
			print(sample_id, end = ", ")
			annots_df_current = annots_df_smp_gp.get_group(sample_id).reset_index(drop = True)
			annots_df_current.set_index('bundle_no', drop = True, inplace = True)
			
			### loop through bundle
			for bundle_no in annots_df_current.index:
				
				#### raw coordinates
				ch = get_heel_coords_sum(bundle_no, annots_df_current)
				ct = get_target_coords_sum(bundle_no, annots_df_current, is_converted = True)
				heel_coords[:,:,ind] = ch
				target_coords[:,:,ind] = ct
				
				#### standardized coordinates
				pos_t3 = annots_df_current.loc[bundle_no, 'T3c']
				pos_t4 = 1
				pos_t7 = annots_df_current.loc[bundle_no, 'T7c']
				dTiCs = {3:pos_t3, 7:pos_t7, 4: pos_t4}
				target_grid_polar = get_target_grid_polar_summary(return_type = 'theory', dTiCs = dTiCs)
				coords_sum[time_id]['R3'][0, ind] = target_grid_polar[2,0]
				coords_sum[time_id]['R3'][1, ind] = annots_df_current.loc[bundle_no, 'R3']
				coords_sum[time_id]['R4'][0, ind] = target_grid_polar[5,0]
				coords_sum[time_id]['R4'][1, ind] = annots_df_current.loc[bundle_no, 'R4']
				coords_sum[time_id]['T3'][0, ind] = target_grid_polar[2,0]
				coords_sum[time_id]['T3'][1, ind] = annots_df_current.loc[bundle_no, 'T3c']
				coords_sum[time_id]['T7'][0, ind] = target_grid_polar[5,0]
				coords_sum[time_id]['T7'][1, ind] = annots_df_current.loc[bundle_no, 'T7c']
				coords_sum[time_id]['T4'][0, ind] = 0
				coords_sum[time_id]['T4'][1, ind] = 1            
				ind += 1
		
		### get centroids
		heels_centroid = get_centroid(heel_coords)
		heel_centroid_sum[:,:,i_fig] = heels_centroid
		target_centroid = get_centroid(target_coords)
		target_centroid_sum[:,:,i_fig] = target_centroid
			
		### plotting
		ax = axes.ravel()[i_fig]
		#### R cells and l cells
		for j in range(6):
			#### R cells
			ax.plot(heels_centroid[j,0], heels_centroid[j,1], 'o', color = color_code[j+1], markersize = 10)
			#### Targets
			ax.plot(target_centroid[j,0], target_centroid[j,1], 'o', mec = color_code[index_to_target_id[j]], markersize = 30, mew = 2, mfc = 'none')
		
		#### reference lines
		ax.plot([0, target_centroid[2,0]], [0, target_centroid[2,1]], '--', color = 'gray', linewidth = 1)
		ax.plot([0, target_centroid[5,0]], [0, target_centroid[5,1]], '--', color = 'gray', linewidth = 1)
		ax.plot([-5, 20], [0,0], '--', color = 'gray')
		
		#### center
		ax.plot(0, 0, 'o', color = 'k', markersize = 10)
		ax.text(0.3, -1, "C")
		
		#### axis
		ax.set_xlim([-4, 16])
		ax.set_ylim([-5, 5])
		ax.set_aspect(aspect=1)
		ax.set_title(f'{time_id} hrs')
		print("")

	fig_name = f'S4A_Fig.{fig_format}'
	fig_save_path = os.path.join(paths.output_prefix, fig_name)
	if(is_save):
		plt.savefig(fig_save_path, dpi=fig_res, bbox_inches='tight', format = fig_format)

	plt.show()

	return heel_centroid_sum, target_centroid_sum, coords_sum

### S4B Fig.


### S4D Fig.

# ================= S6 Fig ================= #
def mutual_repulsion_regression_plot(sum_df, **kwargs):
	"""
	Function: countour plot of regression result vs. actual data for equal and weighted regureesion.

	Inputs:
	- sum_df: DataFrame. processed DataFrame that contains both bundle heel and target info and growth cone length and angle info.
	- Additional inputs:
		- is_save: Boolean. Save figures or not. Default = False.
		- channels: Which channel (or channels) to plot. Default = ["GFP", "RFP"]
		- fig_format: extension figure format. Default = "svg".
		- fig_res: figure resolution. Default = 300.
		- fig_name: name of the figure.

	Output: Figure.
	"""


	### parameters
	if('is_save' in kwargs.keys()):
		is_save = kwargs['is_save']
	else:
		is_save = False
	if('fig_format' in kwargs.keys()):
		fig_format = kwargs['fig_format']
	else:
		fig_format = 'svg'
	if('fig_res' in kwargs.keys()):
		fig_res = kwargs['fig_res']
	else:
		fig_res = 300
	if('fig_name' in kwargs.keys()):
		name = kwargs['fig_name']
		fig_name = f'{name}.{fig_format}'
	else:
		fig_name = f'mutual_repulsion.{fig_format}'	
	paths = settings.paths
	color_code = settings.matching_info.color_code

	### get plotting params
	fig_save_path = os.path.join(paths.output_prefix, fig_name)
	diff_cols = ['ml_diff_theta', 'ml_diff_theta_reg']
	hue = "type_plot"
	
	criteria = (sum_df["time_id"] <= 26) & (sum_df["symmetry"] <= 0.5)
	plot_df_sum = sum_df.loc[criteria, [hue, "time_id"] + diff_cols]
	plot_df_sum[diff_cols] = np.degrees(plot_df_sum[diff_cols])
	plot_df_group = plot_df_sum.groupby("time_id")
	groups = list(plot_df_group.groups.keys())
	
	### figure
	sns.set(font_scale=2)
	sns.set_style("white")
	fig, axes = plt.subplots(2,3, figsize = (20, 12))

	for i in range(6):
		ax = axes.ravel()[i]
		if(i in [0,1,2]):
			diff_col = diff_cols[0]
		elif(i in [3,4,5]):
			diff_col = diff_cols[1]

		### get each timepoint
		pp = plot_df_group.get_group(groups[i%3])

		### R3
		sns.distplot(pp.groupby('type_plot').get_group("R3")[diff_col].dropna(), ax = ax,
					 kde_kws={"color": color_code[3], "ls": '-', 'lw':1.5}, hist = False)
		### R4
		sns.distplot(pp.groupby('type_plot').get_group("R4")[diff_col].dropna(), ax = ax, 
					 kde_kws={"color": color_code[4], "ls": '-', 'lw':1.5}, hist = False)
		### mean
		r3_mean = pp.groupby('type_plot').get_group("R3")[diff_col].mean()
		r4_mean = pp.groupby('type_plot').get_group("R4")[diff_col].mean()
		### reference
		ax.plot([0,0], [0,8], '--', linewidth = 1,color = 'gray', label = '_nolegend_')

		### axis
		ax.set_xlabel("")
		ax.set_ylim([0, 0.05])
		if(i in [0,1,2]):
			ax.set_title(f"{groups[i%3]} hrs")
		ax.set_xlim([-90, 90])
		if(i == 0):
			ax.set_ylabel("Equal Repulsion")
		elif(i == 3):
			ax.set_ylabel("Weighted Repulsion") 
		if(i == 4):
			ax.set_xlabel("Difference In Extension Directions (degrees)")
		if(i not in [0, 3]):
			ax.set_yticks([])
			ax.set_ylabel("")
		else:
			ax.set_xticks([0.00, 0.02, 0.04])
		if(i not in [3, 4, 5]):
			ax.set_xticks([])
		else:
			ax.set_xticks([-90, -45, 0, 45, 90])

	ax.legend(["R3", "R4"], bbox_to_anchor = (1.45, 1.3))
	
	if(is_save):
		plt.savefig(fig_save_path, dpi=fig_res, bbox_inches='tight', format = fig_format, transparent=False)

	plt.show()
