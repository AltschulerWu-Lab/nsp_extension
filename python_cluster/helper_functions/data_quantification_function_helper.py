# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   sf942274
# @Last Modified time: 2019-10-17 21:19:50




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

import data_quantification_settings as settings


"""
Function: Print information to log file
Input: stuff to print
Output:
"""
def print_to_log(info):
	path = settings.paths.log_path
	if(os.path.isfile(path)):
		log_file = open(path, "a+")
	else:
		log_file = open(path, "w+")

	log_file.write(info + '\n')
	log_file.close()



# ================= directory related functions =================
"""
	Function: check if a path exist. If not, make it.
	Input: path
"""
def check_dir(path):
	if not os.path.exists(path):
		if os.path.exists(os.path.dirname(path)):
			os.makedirs(path)
		else:
			check_dir(os.path.dirname(path))

"""
	Function: get the foldres and files within a particular path.
	Input: path
	Output: lists of folders and files
"""
def parse_folder_info(path):
	
	folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	if('.DS_Store' in files):
		files.remove('.DS_Store')
	if('._.DS_Store' in files):
		files.remove('._.DS_Store')
	return folders, files

"""
	Function: get a list of path of every file within a folder (including all its subfolders).
	Input: path
	Output: list of paths.
"""
def get_file_paths(path):
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
'''
	Function: get database column names that have the same pattern (contain or doesn't contain a particular string)
	Input: 
	- df -- dataframe
	- header_tag -- string
	- isContain -- True/False
	Output: list of strings
'''
def group_headers(df, header_tag, isContain):
	
	if isContain:
		return [col for col in df.columns.values if header_tag in col]
	else:
		return [col for col in df.columns.values if header_tag not in col]



# ================= task specific functions =================
def get_target_coords(bundle_No, bundles_df, targetIndexMatch):
	indTs = []
	coordTs = np.zeros((len(targetIndexMatch),2))
	
	coordTs[0,:] = np.array([ bundles_df.loc[bundle_No,'coord_X_T0'], bundles_df.loc[bundle_No,'coord_Y_T0'] ])
	indTs.append(bundle_No)
	
	for i in range(1,len(coordTs)):
		indTs.append(int(bundles_df.loc[bundle_No,'TargetNo_T' + str(targetIndexMatch[i])]))
		coordTs[i,:] = np.array([ bundles_df.loc[indTs[i],'coord_X_T0'], bundles_df.loc[indTs[i],'coord_Y_T0'] ])

	return indTs, coordTs

def get_bundle_center(bundle_No, bundles_df):
	indTs = []
	coord_Center = np.zeros((1,2))
	
	coord_Center[0,:] = np.array([ bundles_df.loc[bundle_No,'coord_X_Center'], bundles_df.loc[bundle_No,'coord_Y_Center'] ])
	# indTs.append(bundle_No)
	
	# for i in range(1,len(coordTs)):
	#   indTs.append(int(bundles_df.loc[bundle_No,'TargetNo_T' + str(targetIndexMatch[i])]))
	#   coordTs[i,:] = np.array([ bundles_df.loc[indTs[i],'coord_X_Center'], bundles_df.loc[indTs[i],'coord_Y_Center'] ])

	return coord_Center

def get_heel_coords(bundle_No, bundles_df):
	HeelCoords = np.zeros((6,2))
	HeelCoords[:,0] = list(bundles_df.loc[bundle_No, my_help.group_headers(bundles_df, 'coord_X_R', True)])
	HeelCoords[:,1] = list(bundles_df.loc[bundle_No, my_help.group_headers(bundles_df, 'coord_Y_R', True)])
	return HeelCoords

def get_rx_coords(bundle_No, bundles_df, indTs, Rtype):
	coordRxs = np.zeros((len(indTs), 2))
	coordHeaders = ['coord_X_R' + str(Rtype), 'coord_Y_R' + str(Rtype)]
	
	for i in range(len(indTs)):
		coordRxs[i,:] = np.array([ bundles_df.loc[ indTs[i], coordHeaders[0]], bundles_df.loc[ indTs[i], coordHeaders[1]]])

	return coordRxs
	
def delete_zero_columns(matrix, factor, axis):
	z_columns = np.unique(np.where(matrix == factor)[axis])
	new_matrix = np.delete(matrix, z_columns, axis = axis)
	return new_matrix

