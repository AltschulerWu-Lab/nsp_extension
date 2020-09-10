# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2020-09-09 04:01:25
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-09 21:00:59

import io, os, sys, types, datetime, pickle, warnings

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import numpy as np
from numpy.linalg import eig, inv
import numpy.ma as ma

import math

from scipy import interpolate, spatial

import scipy.stats as ss
import statsmodels.api as sa
import scikit_posthocs as sp

import skimage.io as skiIo
from skimage import exposure, img_as_float

from sklearn import linear_model, metrics

import settings as settings

# ================= files =================
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

### group DataFrame columns
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


### get heel coordinates
def get_heel_coords_sum(bundle_no, annots_df, **kwargs):
	### unravel params
	dim = 2

	if('is_pixel' in kwargs.keys()):
		is_pixel = kwargs['is_pixel']
	else:
		is_pixel = True
	if('pixel_to_um' in kwargs.keys()):
		pixel_to_um = kwargs['pixel_to_um']
	else:
		pixel_to_um = np.zeros((dim)) + 1
	
	### get heel coordinates
	heel_coords = np.zeros((6,dim))
	heel_coords[:,0] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_X_R', True)])
	heel_coords[:,1] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_Y_R', True)])
	if(not is_pixel):
		for i in range(heel_coords.shape[1]):
			heel_coords[:,i] = heel_coords[:,i] * pixel_to_um[i]

	return heel_coords

### get target coordinates
def get_target_coords_sum(bundle_no, annots_df, **kwargs):
	### unravel params
	dim = 2
	if('return_type' in kwargs.keys()):
		return_type = kwargs['return_type']
	else:
		return_type = 'c'
	if('is_pixel' in kwargs.keys()):
		is_pixel = kwargs['is_pixel']
	else:
		is_pixel = True
	if('pixel_to_um' in kwargs.keys()):
		pixel_to_um = kwargs['pixel_to_um']
	else:
		pixel_to_um = np.zeros((dim)) + 1
	index_to_target_id = settings.matching_info.index_to_target_id

	### get target coordinates
	target_coords = np.zeros((6,dim))
	target_coords[:,0] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_X_T', True)])
	target_coords[:,1] = list(annots_df.loc[bundle_no, group_headers(annots_df, 'coord_Y_T', True)])
	
	return target_coords


def get_angle_unit_theory(return_type):
	### before standardization
	## distance: normal
	dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
	dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
	dT2T5 = dT3T7 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
	dT0T3 = dT0T7 = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5

	## angles: normal
	aT0T2 = math.radians(80)/2
	aT0T5 = - math.radians(80)/2
	aT0T3 = math.acos((dT0T3 ** 2 + dT0T7 ** 2 - dT3T7 ** 2)/(2*dT0T3*dT0T7))/2
	aT0T7 = - aT0T3
	aT0T4 = 0

	## target coordinates
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

	## heel coordinates
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
	# print(aTiCT4)

	aRiCT4 = np.zeros((6,1))
	for i in range(1,6):
		aRiCT4[i] = inner_angle(r_heels_cart[i,:] - c, target_grid_cart[3,:] - c, True)
		if(i in [4,5]):
			aRiCT4[i] = - aRiCT4[i]
	
	### phi_unit
	phi_unit = aTiCT4[2,0]


	if(return_type == 'phi_unit'):
		return phi_unit
	elif(return_type == 'aTiCT4'):
		return aTiCT4
	elif(return_type == 'aRiCT4'):
		return aRiCT4


def get_angle_unit_data(sum_df, **kwargs):
	if('criteria' in kwargs.keys()):
		criteria = kwargs['criteria']
		sum_df = sum_df.loc[criteria, :]
	phi_unit = sum_df['aT3cT7'].mean()/2

	return phi_unit


def get_target_grid_polar_summary(**kwargs):
	index_to_target_id = settings.matching_info.index_to_target_id

	return_type = kwargs['return_type']
	if(return_type == 'theory'):
		dTiCs = kwargs['dTiCs']
		aTiCT4s = get_angle_unit_theory('aTiCT4')
	elif(return_type == 'data'):
		dTiCs = kwargs['dTiCs']
		aTiCT4s =  kwargs['aTiCT4s']

	grid = np.zeros((6,2))
	for i in range(6):
		grid[i,0] = aTiCT4s[i]
		if(index_to_target_id[i] in dTiCs.keys()):
			grid[i,1] = dTiCs[index_to_target_id[i]]

	return grid


def get_heel_grid_polar_summary(**kwargs):
	index_to_target_id = settings.matching_info.index_to_target_id

	return_type = kwargs['return_type']
	if(return_type == 'theory'):
		dRiCs = kwargs['dRiCs']
		aRiCT4 = get_angle_unit_theory('aRiCT4')
	elif(return_type == 'data'):
		dRiCs = kwargs['dRiCs']
		aRiCT4 =  kwargs['aRiCT4']

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


def get_gc_angle_length(sum_df_ri, coord_heels, phi_unit, cat_angle, cat_length, r_type):
	### from standardized coordinate to cartasian coordinate
	target_stc_car, heels_stc_car, gc_tip_car = get_cartasian_grid_from_stc(sum_df_ri, coord_heels, cat_angle, cat_length, phi_unit)

	### get vector of growth cone extension
	if(r_type == 3):
		ori = heels_stc_car[2,:]
	else:
		ori = heels_stc_car[3,:]
	v_gc = gc_tip_car - ori
	
	gc_angle = inner_angle(v_gc, np.array([1,0]), True)
	gc_lengtrh = get_vector_length(v_gc)

	### direction of angle
	if(v_gc[0,1] < 0):
		gc_angle = - gc_angle

	return gc_lengtrh, gc_angle


# ================= mutual repulsion calculation =================
def get_angle_prediction_two_vectors(v1, v2, origin, alphas):
	v1_uni = unit_vector(v1)
	v2_uni = unit_vector(v2)
	
	v_new = alphas[0] * v1_uni + alphas[1] * v2_uni
	point = origin + v_new
	
	v_new = unit_vector(v_new)
	point = origin + v_new
	
	return point, v_new

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


	### repulse from neighbor heels, weighted equally
	alpha = 0.5
	p, v = get_angle_prediction_two_vectors(v1, v2, ori, [alpha, 1-alpha])
	point = np.transpose(p)
	vector = np.transpose(v)
	theta = inner_angle(vector, v_base, True)
	angle = inner_angle(v, np.array([1,0]), True)

	return point, vector, theta, angle, np.vstack((v1, v2))

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


# ================= process summary_df =================
### supporting function: fill sum_df information for each bundle.
def fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR, r_type, phi_unit):
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
	ch = get_heel_coords_sum(bundle_no, annots_df_current)
	ct = get_target_coords_sum(bundle_no, annots_df_current)

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
	
	gc_length, gc_angle = get_gc_angle_length(sum_df.loc[iR,:], ch, phi_unit, cat_angle, cat_length, r_type)
	
	sum_df.loc[iR, f"{cat_angle}_gc"] = gc_angle
	if(r_type == 4):
		sum_df.loc[iR, f"{cat_angle}_gc_mrr"] = 0 - gc_angle
	elif(r_type == 3):
		sum_df.loc[iR, f"{cat_angle}_gc_mrr"] = gc_angle
	sum_df.loc[iR, f"{cat_length}_gc"] = gc_length

	return sum_df


### processing summary_df with calculation of mutual repulsion
def process_sum_df_mr(sum_df_old, annots_df, rel_poses):
	"""
	Function: process summary_df with calculation of mutual repulsion
	Inputs:
	- sum_df_old: DataFrame. Imported angle and length csv file.
	- annots_df: DataFrame. Imported annotation csv file.
	- rel_poses: Dictionary. Relative position info from the image quantification process.
	Output:
	- sum_df: DataFrame. Combined annotation and angle/length info.
	"""
	print("Calculate Mutual Repulsion!")

	### get phi_unit
	criteria = (annots_df['is_Edge'] == 0) & (annots_df['symmetry'] <= 0.5)
	phi_unit = get_angle_unit_data(annots_df, criteria = criteria)

	sum_df = sum_df_old.copy(deep = True)
	paths = settings.paths

	qc_cols = group_headers(annots_df, 'is_', True)

	cols_add = ['heel_pos_type', 'bundle_rcells_total', 'length_fromheel']
	cols_add += qc_cols
	sum_df = dataframe_add_column(sum_df, cols_add)

	### group by time and sample ID
	annots_df_group = annots_df.groupby(['TimeID', 'SampleID'])
	sum_df_group = sum_df.groupby(['TimeID', 'SampleID'])

	### process each sample
	for key in rel_poses.keys():
		timeID = key[0]
		sampleID = key[1]
		rel_pos = rel_poses[key]
		print(f"{timeID}_hrs_sample_{sampleID}", end = ", ")

		if((timeID, sampleID) not in sum_df_group.groups):
			print(f"ERROR! {timeID}hrs_smp{sampleID} not in sum_df!")
		else:
			### sum_df
			sum_df_current = sum_df_group.get_group((timeID, sampleID))
			sum_df_current_gp = sum_df_current.groupby('bundle_no')

			### annots_df
			annots_df_current = annots_df_group.get_group((timeID, sampleID))
			annots_df_current.loc[:,'Bundle_No'] = annots_df_current.loc[:,'Bundle_No'].values.astype(int)
			annots_df_current.set_index('Bundle_No', inplace = True)

			# print(f"bundles numbers: sum_df = {len(sum_df_current_gp)}, annot_df = {len(annots_df_current)}, output = {len(rel_pos)}")

			### process each bundle
			for bundle_no in annots_df_current.index:
				
				### update annotation
				phi_range_1 = rel_pos[bundle_no]["phi_range_1"]
				phi_range_2 = rel_pos[bundle_no]["phi_range_2"]
				symmetry = abs(phi_range_1 - phi_range_2)/max(phi_range_2, phi_range_1)

				##### heel and target grid
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
							
							sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR, r_type, phi_unit)
							
							#### mutual repulsion                                                       
							##### grid in standardized coordinates
							target_stc_car, heels_stc_car, _ = get_cartasian_grid_from_stc(sum_df.loc[iR,:], ch, 'angle', 'length', phi_unit)

							##### get theoretical angles
							point, vector, theta, angle, vs = calculate_mutual_repulsion_theory(heels_stc_car, target_stc_car, r_type)
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
							comp_angle_cols = ['angle_mean_otsu', 'angle_mean_li', 'angle_max_avg', 'angle_max_max']
							comp_length_cols = ['length_mean_otsu', 'length_mean_li', 'length_max', 'length_max']

							cat_angle = 'angle'
							cat_length = 'length'
							gc_point, gc_vector, gc_theta, gc_angle = calculate_mutual_repulsion_data(sum_df.loc[iR,:], ch, phi_unit, cat_angle, cat_length, r_type)

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
						
						sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR3, 3, phi_unit)
						sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR4, 4, phi_unit)
			
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


### processing summary_df without calculation of mutual repulsion
def process_sum_df(sum_df_old, annots_df, rel_poses):
	"""
	Function: process summary_df without calculation of mutual repulsion
	Inputs:
	- sum_df_old: DataFrame. Imported angle and length csv file.
	- annots_df: DataFrame. Imported annotation csv file.
	- rel_poses: Dictionary. Relative position info from the image quantification process.
	Output:
	- sum_df: DataFrame. Combined annotation and angle/length info.
	"""
	print("No mutual repulsion calculation!")

	### get phi_unit
	criteria = (annots_df['is_Edge'] == 0) & (annots_df['symmetry'] <= 0.5)
	phi_unit = get_angle_unit_data(annots_df, criteria = criteria)

	sum_df = sum_df_old.copy(deep = True)
	paths = settings.paths

	qc_cols = group_headers(annots_df, 'is_', True)

	cols_add = ['heel_pos_type', 'bundle_rcells_total', 'length_fromheel']
	cols_add += qc_cols
	sum_df = dataframe_add_column(sum_df, cols_add)

	### group by time and sample ID
	annots_df_group = annots_df.groupby(['TimeID', 'SampleID'])
	sum_df_group = sum_df.groupby(['TimeID', 'SampleID'])

	### process each sample
	for key in rel_poses.keys():
		timeID = key[0]
		sampleID = key[1]
		rel_pos = rel_poses[key]
		print(f"{timeID}_hrs_sample_{sampleID}", end = ", ")

		if((timeID, sampleID) not in sum_df_group.groups):
			print(f"ERROR! {timeID}hrs_smp{sampleID} not in sum_df!")
		else:
			### sum_df
			sum_df_current = sum_df_group.get_group((timeID, sampleID))
			sum_df_current_gp = sum_df_current.groupby('bundle_no')

			### annots_df
			annots_df_current = annots_df_group.get_group((timeID, sampleID))
			annots_df_current.loc[:,'Bundle_No'] = annots_df_current.loc[:,'Bundle_No'].values.astype(int)
			annots_df_current.set_index('Bundle_No', inplace = True)

			# print(f"bundles numbers: sum_df = {len(sum_df_current_gp)}, annot_df = {len(annots_df_current)}, output = {len(rel_pos)}")

			### process each bundle
			for bundle_no in annots_df_current.index:
				
				### update annotation
				phi_range_1 = rel_pos[bundle_no]["phi_range_1"]
				phi_range_2 = rel_pos[bundle_no]["phi_range_2"]
				symmetry = abs(phi_range_1 - phi_range_2)/max(phi_range_2, phi_range_1)

				##### heel and target grid
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
							
							sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR, r_type, phi_unit)
							
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
						
						sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR3, 3, phi_unit)
						sum_df = fill_sum_df_info(sum_df, annots_df_current, rel_pos, num_rcells, bundle_no, iR4, 4, phi_unit)
			

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

# ================= Polar Density Plot =================
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
			 '--', color = 'lightgray', linewidth = 5)
	ax2.plot([0, target_grid_polar['c'][matching_info.target_id_to_index[4],0]], 
			 [0, rlim[1]], 
			 '--', color = 'lightgray', linewidth = 5)
	ax2.plot([0, target_grid_polar['c'][matching_info.target_id_to_index[7],0]], 
			 [0, rlim[1]], 
			 '--', color = 'lightgray', linewidth = 5)
	
	## plot target position
	for i in [0,2,3,5]:
		ax2.plot(target_grid_polar['c'][i,0], target_grid_polar['c'][i,1], 'o', 
				 color = matching_info.color_code[matching_info.index_to_target_id[i]], 
				 markersize = 30, mew = 5, mfc = 'none')


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
		ax2.set_xticklabels(labels = [1, 0, -1])
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

### generate figure.
def generate_density_plot_figure(output_data, **kwargs):
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


# ================= Figure 3B =================
def fit_linear(x, y):
	popt = np.polyfit(x,y,1)
	pred = x * popt[0] + popt[1]
	r2 = get_r_squared(x, y, popt)
	print(f'k = {popt[0]:.2f}, R2 = {r2:.2f}')
	return popt, pred, r2

def get_r_squared(x, y, popt):
	y_fit = popt[0]*x + popt[1]
	residuals = y - y_fit
	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((y-np.mean(y))**2)
	# print(x, y, ss_res, ss_tot)
	r_squared = 1 - (ss_res / ss_tot)
	return r_squared


# ================= Figure 3C =================
def p_value_3c(df, hue_name, value_name, pair_list, print_pair, method):
	df_groups = df.groupby(hue_name)
	for pair in pair_list:
		if(print_pair):
			print(f'==={pair}===')
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
def generate_summary_polar_figure(plot_df, pert_info, **kwargs):
	
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
	theta_cat = 'angle_gc'
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


	# plot reference line
	ax.plot([0,0], [0,2.5], '--', color = 'gray', linewidth = 1)
	ax.plot([0,target_grid_polar[2,0]], [0,2.5], '--', color = 'gray', linewidth = 1)
	ax.plot([0,target_grid_polar[5,0]], [0,2.5], '--', color = 'gray', linewidth = 1)

	# plot targets
	for i in [0,2,3,5]:
		ax.plot(target_grid_polar[i,0], target_grid_polar[i,1], 'o', 
				color = color_code[matching_info.index_to_target_id[i]], markersize = 20, mew = 1.5, mfc = 'none')


	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 8, box.height])

	# axis settings
	ax.set_thetamin(-30)
	ax.set_thetamax(30)
	ax.set_rlim(0, 1.8)
	ax.tick_params(axis = 'x', labelbottom = True)
	ax.tick_params(axis = 'y', labelleft = True)

	plt.yticks(ticks = [0, 0.4, 0.8, 1.2, 1.6])
	plt.xticks(ticks = [-phi_unit, 0, phi_unit], labels = [1, 0, -1])
	ax.set_xlabel("Relative Length (a.u.)")
	ax.yaxis.set_label_position("right")
	plt.ylabel("Relative Angle (a.u.)", labelpad=35)
	plt.title(f"sev>{pert_cat}, {time_id} hrs")

	ax.grid(True)

	if(is_save):
		plt.savefig(fig_save_path, dpi=300, bbox_inches='tight', format = fig_format)
	plt.show()

def p_value_4(df_current, x_cat, y_cat, pert_cat, time_id):
	if(pert_cat == 'Fz'):
		pert_type = 'R3/R3'
	elif(pert_cat == 'Nic'):
		pert_type = 'R4/R4'
		
	if('angle' in y_cat):
		inds = ['R3', f'{pert_type}(3)', 'R3', 'R4']
		cols = ['R4', f'{pert_type}(4)', f'{pert_type}(3)', f'{pert_type}(4)']
	elif('length' in y_cat):
		if(pert_cat == 'Fz'):
			inds = ['R3', f'{pert_type}(3)', 'R3', 'R3']
		elif(pert_cat == 'Nic'):
			inds = ['R3', f'{pert_type}(4)', 'R4', 'R4']
		cols = ['R4', f'{pert_type}(4)', f'{pert_type}(3)', f'{pert_type}(4)']
	
	data = [df_current.loc[ids, y_cat].values for ids in df_current.groupby(x_cat).groups.values()]
	H, p = ss.kruskal(*data)
	df_stat = sp.posthoc_mannwhitney(df_current, val_col=y_cat, group_col=x_cat, p_adjust = 'bonferroni')
	if('length' in y_cat):
		print(f"==={pert_cat}_{time_id}hrs_length===")
		for i in range(len(inds)):
			print(f"{inds[i]} vs {cols[i]}: {df_stat.loc[inds[i], cols[i]]}")
	elif('angle' in y_cat):
		print(f"==={pert_cat}_{time_id}hrs_angle===")
		for i in range(len(inds)):
			print(f"{inds[i]} vs {cols[i]}: {df_stat.loc[inds[i], cols[i]]}")

# ================= Figure S4 =================
def mutual_repulsion_regression_plot(sum_df_ctrl_final, annots_df_ctrl, **kwargs):
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

	
	### regression fitting
	criteria = (sum_df_ctrl_final['symmetry']<=0.5) & (sum_df_ctrl_final['TimeID']<=26)
	sum_df_regression = sum_df_ctrl_final.loc[criteria,:]
	df_regression_results = pd.DataFrame(columns = ['a', 'b', 'r2'])
	print("=== Regression result ===")
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
	
	sum_df_ctrl_group = sum_df_regression.groupby(["TimeID", "SampleID"])
	phi_unit = get_angle_unit_data(annots_df_ctrl, 
											  criteria = (annots_df_ctrl['is_Edge'] == 0) & (annots_df_ctrl['symmetry'] <= 0.5))
	### calculate regression direction
	print("=== Regression direction calculation ===")
	for gp in sum_df_ctrl_group.groups.keys():
		time_id, sample_id = gp
		print(f"{time_id}_hrs_sample_{sample_id}", end = "; ")

		sum_df_current = sum_df_ctrl_group.get_group(gp)
		annots_df_current = annots_df_ctrl.groupby(["TimeID", "SampleID"]).get_group(gp).set_index('Bundle_No')

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

			sum_df_ctrl_final.loc[ind, 'ml_theory_theta_reg'] = theta
			sum_df_ctrl_final.loc[ind, 'ml_theory_angle_reg'] = angle
			sum_df_ctrl_final.loc[ind, 'ml_theory_vec_x_reg'] = v_pred[0]
			sum_df_ctrl_final.loc[ind, 'ml_theory_vec_y_reg'] = v_pred[1]

	for plot_cat in ['angle', 'theta']:
		theory_cat = f"ml_theory_{plot_cat}"
		actual_cat = f"ml_actual_{plot_cat}"
		ref_cat = f"{plot_cat}_ref"
		sum_df_ctrl_final[f"ml_diff_{plot_cat}"] = (sum_df_ctrl_final[theory_cat] - sum_df_ctrl_final[actual_cat])
		
		theory_cat = f"ml_theory_{plot_cat}_reg"
		actual_cat = f"ml_actual_{plot_cat}"
		ref_cat = f"{plot_cat}_ref"
		sum_df_ctrl_final[f"ml_diff_{plot_cat}_reg"] = (sum_df_ctrl_final[theory_cat] - sum_df_ctrl_final[actual_cat])
	
	### get plotting params
	fig_save_path = os.path.join(paths.output_prefix, fig_name)
	diff_cols = ['ml_diff_theta', 'ml_diff_theta_reg']
	hue = "type_plot"
	
	criteria = (sum_df_ctrl_final["TimeID"] <= 26) & (sum_df_ctrl_final["symmetry"] <= 0.5)
	plot_df_sum = sum_df_ctrl_final.loc[criteria, [hue, "TimeID"] + diff_cols]
	plot_df_sum[diff_cols] = np.degrees(plot_df_sum[diff_cols])
	plot_df_group = plot_df_sum.groupby("TimeID")
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
		pp = plot_df_group.get_group(groups[i%3])
		sns.distplot(pp.groupby('type_plot').get_group("R3")[diff_col].dropna(), ax = ax,
					 kde_kws={"color": color_code[3], "ls": '-', 'lw':1.5}, hist = False)
		sns.distplot(pp.groupby('type_plot').get_group("R4")[diff_col].dropna(), ax = ax, 
					 kde_kws={"color": color_code[4], "ls": '-', 'lw':1.5}, hist = False)
		r3_mean = pp.groupby('type_plot').get_group("R3")[diff_col].mean()
		r4_mean = pp.groupby('type_plot').get_group("R4")[diff_col].mean()
		ax.plot([0,0], [0,8], '--', linewidth = 1,color = 'gray', label = '_nolegend_')
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
