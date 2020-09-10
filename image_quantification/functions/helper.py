# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-10 16:08:50


import io, os, sys, types

import pandas as pd


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
import seaborn as sns


import numpy as np
from numpy.linalg import eig, inv
import numpy.ma as ma

import math
from scipy import interpolate, spatial, stats

import skimage.io as skiIo
from skimage import exposure, img_as_float, filters

import settings as settings

# ================= printing =================
def print_to_log(info):
	"""
	Function: Print information to log file and the command line
	Input: stuff to print
	Output:
	"""
	print(info, end = "")
	path = settings.paths.log_path
	if(os.path.isfile(path)):
		log_file = open(path, "a+")
	else:
		log_file = open(path, "w+")

	log_file.write(info)
	log_file.close()



# ================= directory related functions =================
def check_dir(path):
	"""
	Function: check if a path exist. If not, make it.
	Input: path
	"""
	if not os.path.exists(path):
		if os.path.exists(os.path.dirname(path)):
			os.makedirs(path)
		else:
			check_dir(os.path.dirname(path))


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

def get_file_paths(path):	
	"""
	Function: get a list of path of every file within a folder (including all its subfolders).
	Input: path
	Output: list of paths.
	"""
	filePaths = []
	folders, files = parse_folder_info(path)
	if not files:
		for folder in folders:
			paths = get_file_paths(os.path.join(path, folder))
			filePaths = filePaths + paths
	else:
		for file in files:
			filePaths = filePaths + [str(os.path.join(path, file))]
	return filePaths



# ================= dataframe related functions =================

def group_headers(df, header_tag, isContain):
	'''
	Function: get database column names that have the same pattern (contain or doesn't contain a particular string)
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



# ================= getting coordinates from bundles_df =================
def get_targets_info(bundle_no, bundles_df, **kwargs):
	'''
	Function: get index and coordinate information of targets of a given bundle.
	Inputs:
	- bundle_no: int, No. of bundle of interest
	- bundles_df: DataFrame, contains bundles and targets information.
	- kwargs:
		- "return_type": string. return center of targets ("c"), lower ("e1")/higher bound("e2"), or target ellipse ("ellipse")
		- "dim": ind. Dimension of coordinates. 2 or 3.
	Outputs:
	- target_inds: list of ints. No. of targets.
	- target_coords: numpy array. coordinates of targets (or it's major, minor axes and angle.)

	'''
	### unravel params
	if('return_type' in kwargs.keys()):
		return_type = kwargs['return_type']
	else:
		return_type = 'c'
	if('dim' in kwargs.keys()):
		dim = kwargs['dim']
	else:
		dim = 2
	index_to_target_id = settings.matching_info.index_to_target_id
	
	if(return_type == 'ellipse'):
		dim = 3
		coord_cols_list = ['Target_Major', 'Target_Minor', 'Target_Angle']
	else:
		if(dim == 2):
			coord_cols_list = [f'coord_X_T{return_type}', f'coord_Y_T{return_type}']
		elif(dim == 3):
			coord_cols_list = [f'coord_X_T{return_type}', f'coord_Y_T{return_type}', f'coord_Z_Tc']

	target_inds = []
	target_coords = np.zeros((len(index_to_target_id), dim))
	
	for i in range(0,len(target_coords)):
		target_inds.append(int(bundles_df.loc[bundle_no,'TargetNo_T' + str(index_to_target_id[i])]))
		target_coords[i,:] = bundles_df.loc[target_inds[i], coord_cols_list].to_numpy()

	return target_inds, target_coords


def get_bundle_center(bundle_no, bundles_df):
	'''
	Function: get annotated bundle center.
	Inputs:
	- bundle_no: int, No. of bundle of interest
	- bundles_df: DataFrame, contains bundles and targets information.
	Output:
	- center_coord: numpy array. coordinate of annotated bundle center.
	'''
	target_inds = []
	center_coord = np.zeros((1,2))
	
	center_coord[0,:] = np.array([ bundles_df.loc[bundle_no,'coord_X_Center'], bundles_df.loc[bundle_no,'coord_Y_Center'] ])
	
	return center_coord

def get_heel_coords(bundle_no, bundles_df):
	'''
	Function: get heel coordinates.
	Inputs:
	- bundle_no: int, No. of bundle of interest
	- bundles_df: DataFrame, contains bundles and targets information.
	Output:
	- heel_coords: numpy array. coordinate of heels.
	'''
	heel_coords = np.zeros((6,2))
	heel_coords[:,0] = list(bundles_df.loc[bundle_no, group_headers(bundles_df, 'coord_X_R', True)])
	heel_coords[:,1] = list(bundles_df.loc[bundle_no, group_headers(bundles_df, 'coord_Y_R', True)])
	return heel_coords

def get_rx_coords(bundle_no, bundles_df, target_inds, Rtype):
	'''
	Function: get Rx coordinates of the originating bundle and also its putative targets.
	Inputs:
	- bundle_no: int, No. of bundle of interest
	- bundles_df: DataFrame, contains bundles and targets information.
	- target_inds: index of the putative targets of bundle-of-interest
	- Rtype: R-cell type.
	Output:
	- rx_coords: numpy array. coordinates of Rxs.
	'''
	rx_coords = np.zeros((len(target_inds), 2))
	coord_headers = ['coord_X_R' + str(Rtype), 'coord_Y_R' + str(Rtype)]
	rx_coords[0,:] = np.array([ bundles_df.loc[ bundle_no, coord_headers[0]], 
		bundles_df.loc[ bundle_no, coord_headers[1]]])

	for i in range(1, len(target_inds)):
		rx_coords[i,:] = np.array([ bundles_df.loc[ target_inds[i], coord_headers[0]], bundles_df.loc[ target_inds[i], coord_headers[1]]])

	return rx_coords

# ================= others =================
### delete empty columns of a matrix.
def delete_zero_columns(matrix, factor, axis):
	"""
	Function: delete empty columns of a matrix. "empty" is defined by a specific value.
	Inputs:
	- matrix: numpy array (n-dimentional).
	- factor: float. value defined as empty.
	- axis: axis of matrix to delete.
	Output:
	- new_matrix: numpy array (n-dimentional). matrix with empty columns deleted.
	"""
	
	z_columns = np.unique(np.where(matrix == factor)[axis])
	new_matrix = np.delete(matrix, z_columns, axis = axis)
	return new_matrix

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


