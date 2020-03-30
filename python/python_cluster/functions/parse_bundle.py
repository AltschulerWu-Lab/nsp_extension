# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-03-27 15:07:39


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

### get bundle information Version 1: 8 points per ROI. Center of bundle as last and target position as second to last.
"""
Function: consolidate bundle information.
Input: 
- roi_df, annot_df: dataframes contain ROI information from imageJ output and annotation information from csv file.
- x_ratio, y_ratio: from microns to pixels
- is_extended_target_list: only T2~T7 or T1~T7.
Output: bundles_df: dataframe with bundle information.
- columns: 'AgeID', 'CategoryID', 'Orientation_AP', 'Orientation_DV', 'RegionID',
	   'SampleID', 'coord_X_Center', 'coord_X_R1', 'coord_X_R2', 'coord_X_R3',
	   'coord_X_R4', 'coord_X_R5', 'coord_X_R6', 'coord_X_T0',
	   'coord_Y_Center', 'coord_Y_R1', 'coord_Y_R2', 'coord_Y_R3',
	   'coord_Y_R4', 'coord_Y_R5', 'coord_Y_R6', 'coord_Y_T0',
	   'coord_Z_Center', 'coord_Z_R1', 'coord_Z_R2', 'coord_Z_R3',
	   'coord_Z_R4', 'coord_Z_R5', 'coord_Z_R6', 'coord_Z_T0', 'is_Crowded',
	   'is_Edge', 'is_Heel_Issues', 'is_Mirror_Symmetry', 'is_symFate',
	   'num_Rcells', 'TargetNo_T3', 'TargetNo_T4', 'TargetNo_T5',
	   'TargetNo_T2', 'TargetNo_T7'
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
	bundles_cols = ['Bundle_No', 'num_Rcells'] + r_coords_list + center_coords_list + qc_col_names + orient_col_names + img_col_names + flag_col_names
	
	### create new dataframe
	bundles_df = pd.DataFrame(columns = bundles_cols)
	
	### get ROI data grouped by label
	roi_df_group = roi_df.groupby('Label').mean()
	roi_df_group.sort_values('No', inplace=True)
	roi_df_group.reset_index(inplace=True)
	
	### update bundle coordinates
	# print("---bundles_df---")
	for ind in roi_df_group.index:
		df_tmp = pd.DataFrame(columns = bundles_cols)
		df_tmp.loc[0,'Bundle_No'] = int(ind+1)
		df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]
		df_tmp.loc[0,'num_Rcells'] = int(df_bd.shape[0])
		
		# error: bundle annotation from imageJ doesn't have enough information
		if(len(df_bd) != 8):
			print('Error! Bundle No. ' + str(ind+1) + 'ROI count inaccurate!')
		
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
		bundles_df = bundles_df.append(df_tmp, ignore_index=True, sort=True)
		
	# set bundle no as index
	bundles_df = bundles_df.set_index('Bundle_No')  
	
	
	### update target & quality-control info
	print("---annot_df---")
	my_help.print_to_log("---annot_df---")

	for ind in annot_df.index:
		bundle_no = annot_df.iloc[ind]['Bundle_No'].astype(int)
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

### get bundle information Version 2: 8 points per ROI. Target position as last and center of bundle as second to last.
"""
Function: consolidate bundle information.
Input: 
- roi_df, annot_df: dataframes contain ROI information from imageJ output and annotation information from csv file.
- x_ratio, y_ratio: from microns to pixels
- is_extended_target_list: only T2~T7 or T1~T7.
Output: bundles_df: dataframe with bundle information.
- columns: 'AgeID', 'CategoryID', 'Orientation_AP', 'Orientation_DV', 'RegionID',
	   'SampleID', 'coord_X_Center', 'coord_X_R1', 'coord_X_R2', 'coord_X_R3',
	   'coord_X_R4', 'coord_X_R5', 'coord_X_R6', 'coord_X_T0',
	   'coord_Y_Center', 'coord_Y_R1', 'coord_Y_R2', 'coord_Y_R3',
	   'coord_Y_R4', 'coord_Y_R5', 'coord_Y_R6', 'coord_Y_T0',
	   'coord_Z_Center', 'coord_Z_R1', 'coord_Z_R2', 'coord_Z_R3',
	   'coord_Z_R4', 'coord_Z_R5', 'coord_Z_R6', 'coord_Z_T0', 'is_Crowded',
	   'is_Edge', 'is_Heel_Issues', 'is_Mirror_Symmetry', 'is_symFate',
	   'num_Rcells', 'TargetNo_T3', 'TargetNo_T4', 'TargetNo_T5',
	   'TargetNo_T2', 'TargetNo_T7'
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
	bundles_cols = ['Bundle_No', 'num_Rcells'] + r_coords_list + center_coords_list + qc_col_names + orient_col_names + img_col_names + flag_col_names
	
	### create new dataframe
	bundles_df = pd.DataFrame(columns = bundles_cols)
	
	### get ROI data grouped by label
	roi_df_group = roi_df.groupby('Label').mean()
	roi_df_group.sort_values('No', inplace=True)
	roi_df_group.reset_index(inplace=True)
	
	### update bundle coordinates
	for ind in roi_df_group.index:
		df_tmp = pd.DataFrame(columns = bundles_cols)
		df_tmp.loc[0,'Bundle_No'] = int(ind+1)
		df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]
		df_tmp.loc[0,'num_Rcells'] = int(df_bd.shape[0])
		
		# error: bundle annotation from imageJ doesn't have enough information
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
		
		bundles_df = bundles_df.append(df_tmp, ignore_index=True, sort=True)
		
	## set Bundle_No as index
	bundles_df = bundles_df.set_index('Bundle_No')  
	
	
	### update target & quality-control info
	print("---annot_df---")
	my_help.print_to_log("---annot_df---")

	for ind in annot_df.index:
		
		bundle_no = annot_df.iloc[ind]['Bundle_No'].astype(int)
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


### get bundle information Version 3: 7 or 1 point(s) per ROI. Heel bundle w/ R1-R6 and bundle center vs. target bundle w/ only target position.
"""
Function: consolidate bundle information.
Input: 
- roi_df, annot_df: dataframes contain ROI information from imageJ output and annotation information from csv file.
- x_ratio, y_ratio: from microns to pixels
- is_extended_target_list: only T2~T7 or T1~T7.
Output: bundles_df: dataframe with bundle information.
- columns: 'AgeID', 'CategoryID', 'Orientation_AP', 'Orientation_DV', 'RegionID',
	   'SampleID', 'coord_X_Center', 'coord_X_R1', 'coord_X_R2', 'coord_X_R3',
	   'coord_X_R4', 'coord_X_R5', 'coord_X_R6', 'coord_X_T0',
	   'coord_Y_Center', 'coord_Y_R1', 'coord_Y_R2', 'coord_Y_R3',
	   'coord_Y_R4', 'coord_Y_R5', 'coord_Y_R6', 'coord_Y_T0',
	   'coord_Z_Center', 'coord_Z_R1', 'coord_Z_R2', 'coord_Z_R3',
	   'coord_Z_R4', 'coord_Z_R5', 'coord_Z_R6', 'coord_Z_T0', 'is_Crowded',
	   'is_Edge', 'is_Heel_Issues', 'is_Mirror_Symmetry', 'is_symFate',
	   'num_Rcells', 'TargetNo_T3', 'TargetNo_T4', 'TargetNo_T5',
	   'TargetNo_T2', 'TargetNo_T7'
"""
def get_bundles_info_v3(roi_df, annot_df, x_ratio, y_ratio, is_extended_target_list):
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
    bundles_cols = ['Bundle_No', 'num_Rcells'] + r_coords_list + center_coords_list + qc_col_names + orient_col_names + img_col_names + flag_col_names

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

        ## this ROI is a bundle
        if(roi_df_group.loc[ind, 'count'] == 7):
            df_tmp = pd.DataFrame(columns = bundles_cols)
            df_tmp.loc[0,'Bundle_No'] = int(ind+1)
            df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]
            df_tmp.loc[0,'num_Rcells'] = int(df_bd.shape[0])

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
            df_tmp = pd.DataFrame(columns = bundles_cols)
            df_tmp.loc[0,'Bundle_No'] = int(ind+1)
            df_bd = roi_df.loc[roi_df['Label'] == list(roi_df_group.Label)[ind]]

            ## target coordinates
            df_tmp.loc[0,'coord_X_T0'] = float(df_bd.loc[df_bd.index[0], 'X'])
            df_tmp.loc[0,'coord_Y_T0'] = float(df_bd.loc[df_bd.index[0], 'Y'])
            df_tmp.loc[0,'coord_Z_T0'] = float(df_bd.loc[df_bd.index[0], 'Slice'])

            bundles_df = bundles_df.append(df_tmp, ignore_index=True, sort=True)

        else:
            print(f'ERROR! Bundle No. {ind+1} count incorrect!')

    bundles_df = bundles_df.set_index('Bundle_No')

    ### update target & quality-control info
    print("---annot_df---")
    my_help.print_to_log("---annot_df---")

    for ind in annot_df.index:

        bundle_no = annot_df.iloc[ind]['Bundle_No'].astype(int)
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

        ### T0 coordinates
        ind_t0 = annot_df.loc[ind,'T0'].astype(int)
        bundles_df.loc[bundle_no, 'coord_X_T0'] = bundles_df.loc[ind_t0, 'coord_X_T0']
        bundles_df.loc[bundle_no, 'coord_Y_T0'] = bundles_df.loc[ind_t0, 'coord_Y_T0']
        bundles_df.loc[bundle_no, 'coord_Z_T0'] = bundles_df.loc[ind_t0, 'coord_Z_T0']

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