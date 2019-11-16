# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   sf942274
# @Last Modified time: 2019-09-26 16:48:54


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

import data_quantification_function_helper as my_help
import data_quantification_settings as settings

""" ============== Parsing bundle information =============="""
def good_qc_df(bundles_df):
	qc_cols = my_help.group_headers(bundles_df, 'is_', True)
	conf_index = [False] * len(bundles_df.index)
	for col in qc_cols:
		conf_index = conf_index | (bundles_df.loc[:,col] == 1)
	return bundles_df[~conf_index]

### bad quality bundles
def bad_qc_df(bundles_df):
	qc_cols = my_help.group_headers(bundles_df, 'is_', True)
	conf_index = [False] * len(bundles_df.index)
	for col in qc_cols:
		conf_index = conf_index | (bundles_df.loc[:,col] == 1)
	return bundles_df[conf_index]

"""
Version 1:
Center of bundle as last and target position as second to last.
"""
def get_bundles_info_v1(roi_df, annot_df, x_ratio, y_ratio, is_extended_target_list):
	### initialization
	r_coords_list = []
	for i in range(6):
		for j in ['X', 'Y', 'Z']:
			r_coords_list.append('coord' + '_' + j + '_R' + str(i+1))
	center_coords_list = []
	for j in ['X', 'Y', 'Z']:
		center_coords_list.append('coord_' + j + '_Center')
	
	### get column names
	qc_col_names = my_help.group_headers(annot_df, 'is_', True)
	orient_col_names = my_help.group_headers(annot_df, 'Orientation_', True)
	img_col_names = my_help.group_headers(annot_df, 'ID', True)
	flag_col_names = my_help.group_headers(annot_df, 'if_', True)
	bundles_cols = ['bundle_no', 'num_Rcells'] + r_coords_list + center_coords_list + qc_col_names + orient_col_names + img_col_names + flag_col_names
	
	### create new dataframe
	bundles_df = pd.DataFrame(columns = bundles_cols)
	
	### get ROI data grouped by label
	roi_df_group = roi_df.groupby('Label').mean()
	roi_df_group.sort_values('No', inplace=True)
	roi_df_group.reset_index(inplace=True)
	
	### update bundle coordinates
	# print("---bundles_df---")
	for ind in roi_df_group.index:
		# print(ind)
		df_tmp = pd.DataFrame(columns = bundles_cols)
		df_tmp.loc[0,'bundle_no'] = int(ind+1)
		df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]
		df_tmp.loc[0,'num_Rcells'] = int(df_bd.shape[0])
		if(len(df_bd) != 8):
			print('Bundle No. ' + str(ind+1) + 'ROI count inaccurate!')
		
		if(len(df_bd) == 8):
			## R1- R6
			for i in range(6):
				df_tmp.loc[0,['coord_X_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'X'])
				df_tmp.loc[0,['coord_Y_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'Y'])
				df_tmp.loc[0,['coord_Z_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'Slice'])
			
			## target
			df_tmp.loc[0,'coord_X_T0'] = float(df_bd.loc[df_bd.index[6], 'X'])
			df_tmp.loc[0,'coord_Y_T0'] = float(df_bd.loc[df_bd.index[6], 'Y'])
			df_tmp.loc[0,'coord_Z_T0'] = float(df_bd.loc[df_bd.index[6], 'Slice'])

			## center of bundle
			df_tmp.loc[0,'coord_X_Center'] = float(df_bd.loc[df_bd.index[7], 'X'])
			df_tmp.loc[0,'coord_Y_Center'] = float(df_bd.loc[df_bd.index[7], 'Y'])
			df_tmp.loc[0,'coord_Z_Center'] = float(df_bd.loc[df_bd.index[7], 'Slice'])
		bundles_df = bundles_df.append(df_tmp, ignore_index=True)
		
	# set bundle no as index
	bundles_df = bundles_df.set_index('bundle_no')  
	
	
	### update target & quality-control info
	print("---annot_df---")
	my_help.print_to_log("---annot_df---")

	for ind, bundle_no in enumerate(annot_df.index):
	# for ind in annot_df.index:
		# bundle_no = annot_df.iloc[ind]['bundle_no'].astype(int)
		print(bundle_no)
		my_help.print_to_log(str(bundle_no))

		### target info
		if(is_extended_target_list):
			bundles_df.loc[bundle_no, 'TargetNo_T3'] = annot_df.loc[ind,'T3'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T4'] = annot_df.loc[ind,'T4'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T5'] = annot_df.loc[ind,'T5'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T2'] = annot_df.loc[ind,'T2'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T1'] = annot_df.loc[ind,'T1'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T6'] = annot_df.loc[ind,'T6'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T20'] = annot_df.loc[ind,'T20'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T50'] = annot_df.loc[ind,'T50'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T40'] = annot_df.loc[ind,'T40'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T30'] = annot_df.loc[ind,'T30'].astype(int)
		else:
			bundles_df.loc[bundle_no, 'TargetNo_T3'] = annot_df.loc[ind,'T3'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T4'] = annot_df.loc[ind,'T4'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T5'] = annot_df.loc[ind,'T5'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T2'] = annot_df.loc[ind,'T2'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T7'] = annot_df.loc[ind,'T7'].astype(int)
		### quality control and flag info
		bundles_df.loc[bundle_no, orient_col_names] = annot_df.loc[ind, orient_col_names]    
		bundles_df.loc[bundle_no, qc_col_names] = annot_df.loc[ind, qc_col_names].astype(int)
		bundles_df.loc[bundle_no, flag_col_names] = annot_df.loc[ind, flag_col_names].astype(int)
		
		### category, sample and region info.
		bundles_df.loc[bundle_no, img_col_names] = annot_df.loc[ind, img_col_names] 
	
	### X,Y coordinates from microns to pixels
	x_coord_cols = my_help.group_headers(bundles_df, 'coord_X', True)
	bundles_df.loc[:,x_coord_cols] = bundles_df.loc[:,x_coord_cols] * x_ratio
	y_coord_cols = my_help.group_headers(bundles_df, 'coord_Y', True)
	bundles_df.loc[:,y_coord_cols] = bundles_df.loc[:,y_coord_cols] * y_ratio
	
	return bundles_df

"""
Version:
Target position as last and center of bundle as second to last.
"""
def get_bundles_info_v2(roi_df, annot_df, x_ratio, y_ratio, is_extended_target_list):
	### initialization
	r_coords_list = []
	for i in range(6):
		for j in ['X', 'Y', 'Z']:
			r_coords_list.append('coord' + '_' + j + '_R' + str(i+1))
	center_coords_list = []
	for j in ['X', 'Y', 'Z']:
		center_coords_list.append('coord_' + j + '_Center')
	
	### get column names
	qc_col_names = my_help.group_headers(annot_df, 'is_', True)
	orient_col_names = my_help.group_headers(annot_df, 'Orientation_', True)
	img_col_names = my_help.group_headers(annot_df, 'ID', True)
	flag_col_names = my_help.group_headers(annot_df, 'if_', True)
	bundles_cols = ['bundle_no', 'num_Rcells'] + r_coords_list + center_coords_list + qc_col_names + orient_col_names + img_col_names + flag_col_names
	
	### create new dataframe
	bundles_df = pd.DataFrame(columns = bundles_cols)
	
	### get ROI data grouped by label
	roi_df_group = roi_df.groupby('Label').mean()
	roi_df_group.sort_values('No', inplace=True)
	roi_df_group.reset_index(inplace=True)
	
	### update bundle coordinates
	# print("---bundles_df---")
	for ind in roi_df_group.index:
		# print(ind)
		df_tmp = pd.DataFrame(columns = bundles_cols)
		df_tmp.loc[0,'bundle_no'] = int(ind+1)
		df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]
		df_tmp.loc[0,'num_Rcells'] = int(df_bd.shape[0])
		print(len(df_bd))
		if(len(df_bd) != 8):
			print('Bundle No. ' + str(ind+1) + 'ROI count inaccurate!')
		
		if(len(df_bd) == 8):
			## R1- R6
			for i in range(6):
				df_tmp.loc[0,['coord_X_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'X'])
				df_tmp.loc[0,['coord_Y_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'Y'])
				df_tmp.loc[0,['coord_Z_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'Slice'])
			
			## target
			df_tmp.loc[0,'coord_X_T0'] = float(df_bd.loc[df_bd.index[7], 'X'])
			df_tmp.loc[0,'coord_Y_T0'] = float(df_bd.loc[df_bd.index[7], 'Y'])
			df_tmp.loc[0,'coord_Z_T0'] = float(df_bd.loc[df_bd.index[7], 'Slice'])

			## center of bundle
			df_tmp.loc[0,'coord_X_Center'] = float(df_bd.loc[df_bd.index[6], 'X'])
			df_tmp.loc[0,'coord_Y_Center'] = float(df_bd.loc[df_bd.index[6], 'Y'])
			df_tmp.loc[0,'coord_Z_Center'] = float(df_bd.loc[df_bd.index[6], 'Slice'])
		
		bundles_df = bundles_df.append(df_tmp, ignore_index=True)
		
	# set bundle no as index
	bundles_df = bundles_df.set_index('bundle_no')  
	
	
	### update target & quality-control info
	print("---annot_df---")
	my_help.print_to_log("---annot_df---")

	for ind in annot_df.index:
		
		bundle_no = annot_df.iloc[ind]['bundle_no'].astype(int)
		print(bundle_no)
		my_help.print_to_log(str(bundle_no))
		
		### target info
		if(is_extended_target_list):
			bundles_df.loc[bundle_no, 'TargetNo_T3'] = annot_df.loc[ind,'T3'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T4'] = annot_df.loc[ind,'T4'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T5'] = annot_df.loc[ind,'T5'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T2'] = annot_df.loc[ind,'T2'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T1'] = annot_df.loc[ind,'T1'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T6'] = annot_df.loc[ind,'T6'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T20'] = annot_df.loc[ind,'T20'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T50'] = annot_df.loc[ind,'T50'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T40'] = annot_df.loc[ind,'T40'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T30'] = annot_df.loc[ind,'T30'].astype(int)
		else:
			bundles_df.loc[bundle_no, 'TargetNo_T3'] = annot_df.loc[ind,'T3'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T4'] = annot_df.loc[ind,'T4'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T5'] = annot_df.loc[ind,'T5'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T2'] = annot_df.loc[ind,'T2'].astype(int)
			bundles_df.loc[bundle_no, 'TargetNo_T7'] = annot_df.loc[ind,'T7'].astype(int)
		### quality control and flag info
		bundles_df.loc[bundle_no, orient_col_names] = annot_df.loc[ind, orient_col_names]    
		bundles_df.loc[bundle_no, qc_col_names] = annot_df.loc[ind, qc_col_names].astype(int)
		bundles_df.loc[bundle_no, flag_col_names] = annot_df.loc[ind, flag_col_names].astype(int)
		
		### category, sample and region info.
		bundles_df.loc[bundle_no, img_col_names] = annot_df.loc[ind, img_col_names] 
	
	### X,Y coordinates from microns to pixels
	x_coord_cols = my_help.group_headers(bundles_df, 'coord_X', True)
	bundles_df.loc[:,x_coord_cols] = bundles_df.loc[:,x_coord_cols] * x_ratio
	y_coord_cols = my_help.group_headers(bundles_df, 'coord_Y', True)
	bundles_df.loc[:,y_coord_cols] = bundles_df.loc[:,y_coord_cols] * y_ratio
	
	return bundles_df