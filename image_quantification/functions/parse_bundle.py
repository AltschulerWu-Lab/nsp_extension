# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-10 16:13:43


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


import helper as my_help
import settings as settings

### get bundles with good qc score
"""
Function: get bundles with good qc score
Input: bundles_df
Output: subset of bundles_df with good qc score
"""
def good_qc_df(bundles_df):
	qc_cols = my_help.group_headers(bundles_df, 'is_', True)
	conf_index = [False] * len(bundles_df.index)
	for col in qc_cols:
		conf_index = conf_index | (bundles_df.loc[:,col] == 1)
	return bundles_df[~conf_index]

### get bundles with bad qc score
"""
Function: get bundles with bad qc score
Input: bundles_df
Output: subset of bundles_df with bad qc score
"""
def bad_qc_df(bundles_df):
	qc_cols = my_help.group_headers(bundles_df, 'is_', True)
	conf_index = [False] * len(bundles_df.index)
	for col in qc_cols:
		conf_index = conf_index | (bundles_df.loc[:,col] == 1)
	return bundles_df[conf_index]


### get bundle information: ROI as 7 points (heels) or 1 ellipse (target)
### 
"""
Function: consolidate bundle information.
Input: 
- roi_df, annot_df: dataframes contain ROI information from imageJ output and annotation information from csv file.
- x_ratio, y_ratio: from microns to pixels
- **kwargs: additional info
	- "is_extended_target_list": Boolean. True: T0 ~ T7; False: T0 + T2 ~ T5 + T7
	- "is_print": Boolean. True: print more info; False: print less info.
Output: bundles_df: dataframe with bundle information.
- columns: 'Bundle_Type', 'CategoryID', 'Orientation_AP', 'Orientation_DV',
       'RegionID', 'SampleID', 'Target_Angle', 'Target_Major', 'Target_Minor',
       'TimeID', 'coord_X_Center', 'coord_X_R1', 'coord_X_R2', 'coord_X_R3',
       'coord_X_R4', 'coord_X_R5', 'coord_X_R6', 'coord_X_Tc', 'coord_X_Te1',
       'coord_X_Te2', 'coord_Y_Center', 'coord_Y_R1', 'coord_Y_R2',
       'coord_Y_R3', 'coord_Y_R4', 'coord_Y_R5', 'coord_Y_R6', 'coord_Y_Tc',
       'coord_Y_Te1', 'coord_Y_Te2', 'coord_Z_Center', 'coord_Z_R1',
       'coord_Z_R2', 'coord_Z_R3', 'coord_Z_R4', 'coord_Z_R5', 'coord_Z_R6',
       'coord_Z_Tc', 'is_Crowded', 'is_Edge', 'is_Heel_Issues',
       'is_Irregular_Grid', 'is_Mirror_Symmetry', 'is_Weak_FasII',
       'is_symFate', 'TargetNo_T0', 'TargetNo_T3', 'TargetNo_T4',
       'TargetNo_T5', 'TargetNo_T2', 'TargetNo_T7'
"""
def get_bundles_info(roi_df, annot_df, x_ratio, y_ratio, **kwargs):
	### unravel kwargs
	if('is_print' in kwargs.keys()):
		is_print = kwargs['is_print']
	else:
		is_print = False
	if('is_extended_target_list' in kwargs.keys()):
		is_extended_target_list = kwargs['is_extended_target_list']
	else:
		is_extended_target_list = False


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
	bundles_cols = ['Bundle_Type', 'Bundle_No'] + r_coords_list + center_coords_list + qc_col_names + orient_col_names + img_col_names + flag_col_names
		
	### create new dataframe
	bundles_df = pd.DataFrame(columns = bundles_cols)

	### group ROI.csv according to label -- grouping individual roi together
	roi_df_group = roi_df.groupby('Label') \
		.agg({'X':'size', 'No':'mean'}) \
		.rename(columns={'X':'count','No':'order'})
	roi_df_group.sort_values('order', inplace=True) #sort according to order of roi added, so that ind of roi_df_group = bundle_no - 1
	roi_df_group.reset_index(inplace=True)

	### update bundle coordinates
	for ind in roi_df_group.index:
		bundle_no = int(ind+1)
		if(is_print):
			print(bundle_no, roi_df_group.loc[ind, 'count'])

		## this ROI is a bundle
		if(roi_df_group.loc[ind, 'count'] == 7):
			if(is_print):
				print(f'{bundle_no}: bundle!')
			df_tmp = pd.DataFrame(columns = bundles_cols)
			df_tmp.loc[0,'Bundle_Type'] = 'Heel'
			df_tmp.loc[0,'Bundle_No'] = bundle_no
			df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]
			# df_tmp.loc[0,'num_Rcells'] = int(df_bd.shape[0])

			## R1- R6 coordinates
			for i in range(6):
				df_tmp.loc[0,['coord_X_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'X'])
				df_tmp.loc[0,['coord_Y_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'Y'])
				df_tmp.loc[0,['coord_Z_R' + str(i+1)]] = float(df_bd.loc[df_bd.index[i], 'Slice'])

			## center of bundle coordinates
			df_tmp.loc[0,'coord_X_Center'] = float(df_bd.loc[df_bd.index[6], 'X'])
			df_tmp.loc[0,'coord_Y_Center'] = float(df_bd.loc[df_bd.index[6], 'Y'])
			df_tmp.loc[0,'coord_Z_Center'] = float(df_bd.loc[df_bd.index[6], 'Slice'])

			bundles_df = bundles_df.append(df_tmp, ignore_index=True, sort=True)

		## this ROI is a target
		elif(roi_df_group.loc[ind, 'count'] == 1):
			if(is_print):
				print(f'{bundle_no}: target !')
			df_tmp = pd.DataFrame(columns = bundles_cols)
			df_tmp.loc[0,'Bundle_No'] = bundle_no
			df_tmp.loc[0,'Bundle_Type'] = 'Target'
			df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]

			## ellipse params
			df_tmp.loc[0,'Target_Major'] = df_bd.loc[df_bd.index[0],"Major"]
			df_tmp.loc[0,'Target_Minor'] = df_bd.loc[df_bd.index[0],"Minor"]
			df_tmp.loc[0,'Target_Angle'] = 180 - df_bd.loc[df_bd.index[0],"Angle"]
						
			## target coordinates: center
			df_tmp.loc[0,'coord_X_Tc'] = float(df_bd.loc[df_bd.index[0], 'X'])
			df_tmp.loc[0,'coord_Y_Tc'] = float(df_bd.loc[df_bd.index[0], 'Y'])
			df_tmp.loc[0,'coord_Z_Tc'] = float(df_bd.loc[df_bd.index[0], 'Slice'])

			## target coordinates: extended
			df_tmp.loc[0,'coord_X_Te1'] = df_tmp.loc[0,'coord_X_Tc'] - df_tmp.loc[0,'Target_Major']*0.5*np.cos(np.radians(df_tmp.loc[0,'Target_Angle']))
			df_tmp.loc[0,'coord_Y_Te1'] = df_tmp.loc[0,'coord_Y_Tc'] - df_tmp.loc[0,'Target_Major']*0.5*np.sin(np.radians(df_tmp.loc[0,'Target_Angle']))
			df_tmp.loc[0,'coord_X_Te2'] = df_tmp.loc[0,'coord_X_Tc'] + df_tmp.loc[0,'Target_Major']*0.5*np.cos(np.radians(df_tmp.loc[0,'Target_Angle']))
			df_tmp.loc[0,'coord_Y_Te2'] = df_tmp.loc[0,'coord_Y_Tc'] + df_tmp.loc[0,'Target_Major']*0.5*np.sin(np.radians(df_tmp.loc[0,'Target_Angle']))

			bundles_df = bundles_df.append(df_tmp, ignore_index=True, sort=True)

		## count of this ROI incorrect:
		else:
			print(f'ERROR!! ROI no {ind+1} numbers not correct!')

	bundles_df = bundles_df.set_index('Bundle_No')
	

	## update target & quality-control info
	for ind in annot_df.index:

		bundle_no = annot_df.iloc[ind]['Bundle_No'].astype(int)
		my_help.print_to_log(f'{bundle_no}, ')

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
			bundles_df.loc[bundle_no, 'TargetNo_T0'] = annot_df.loc[ind,'T0'].astype(int)
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
	bundles_df.loc[:,'Target_Major'] = bundles_df.loc[:,'Target_Major'] * x_ratio
	bundles_df.loc[:,'Target_Minor'] = bundles_df.loc[:,'Target_Minor'] * x_ratio

	my_help.print_to_log("\n")
	return bundles_df