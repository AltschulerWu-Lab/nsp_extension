# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2018-10-19 00:59:49
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-04-01 06:22:51


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
    if((-np.pi <= phi_end <= 0) & (phi_end*phi_start < 0) & (phi_end < angle_normalization(phi_start - np.pi))):
        phi_end_transform = phi_end + 2*np.pi
        phis = np.linspace(phi_start, phi_end_transform, num_of_slices)
    elif((-np.pi <= phi_start <= -0.5*np.pi) & (phi_end*phi_start < 0) & (phi_end < angle_normalization(phi_start - np.pi))):
        phi_start_transform = phi_start + 2*np.pi
        phis = np.linspace(phi_end, phi_start_transform, num_of_slices)
    else:
        phis = np.linspace(phi_start, phi_end, num_of_slices)
    phis_final = angle_normalization(phis)
    return phis_final

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
    if(((-np.pi <= ang_start2r <= -0.5*np.pi) | (-np.pi <= ang_end2r <= -0.5*np.pi)) & (ang_start2r*ang_end2r < 1) ):
        if((-np.pi <= ang_start2r <= -0.5*np.pi) & (-np.pi <= ang_end2r <= -0.5*np.pi)):
            phi_start = min(ang_start2r, ang_end2r) - num_outside_angle * phi_unit
            phi_end = max(ang_start2r, ang_end2r) + num_outside_angle * phi_unit
        else:
            phi_start = max(ang_start2r, ang_end2r) - num_outside_angle * phi_unit
            phi_end = min(ang_start2r, ang_end2r) + num_outside_angle * phi_unit
    else:
        phi_start = min(ang_start2r, ang_end2r) - num_outside_angle * phi_unit
        phi_end = max(ang_start2r, ang_end2r) + num_outside_angle * phi_unit

    phi_start = angle_normalization(phi_start)
    phi_end = angle_normalization(phi_end)
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
    phis = get_phis(phi_start, phi_end, analysis_params_general.num_angle_section + analysis_params_general.num_outside_angle * 2 + 1)

    # aligh angle slices to from T7 to T3.
    if(smallest_angle(ang_start2r, phis[-1]) < smallest_angle(ang_start2r, phis[0])):
        phis = np.flip(phis, axis = 0)
    
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
    
    phi_start_1, phi_end_1 = get_start_end(ang_negone2r, ang_zero, analysis_params_general.num_outside_angle, phi_unit_1)


    phis_1 = get_phis(phi_start_1, phi_end_1, angle_sel_num + analysis_params_general.num_outside_angle*2 + 1)

    if(smallest_angle(ang_negone2r, phis_1[-1]) < smallest_angle(ang_negone2r, phis_1[0])):
        phis_1 = np.flip(phis_1, axis = 0)

    n_1 = int(angle_sel_num + analysis_params_general.num_outside_angle + 1)
    
    phis_1 = phis_1[0:n_1]
    phis_1[n_1-1] = ang_zero

    y_ticks_1 = np.linspace(-1 - analysis_params_general.num_outside_angle*phi_unit_1/phi_range_1, 0, angle_sel_num + analysis_params_general.num_outside_angle + 1)


    ## middle ~ T3 (0 ~ 1)
    phi_range_2 = inner_angle(slice_one_point - center_point, target_coords[3] - center_point, True)
    phi_unit_2 = phi_range_2/angle_sel_num

    phi_start_2, phi_end_2 = get_start_end(ang_zero, ang_one2r, analysis_params_general.num_outside_angle, phi_unit_2)

    phis_2 = get_phis(phi_start_2, phi_end_2, angle_sel_num + analysis_params_general.num_outside_angle*2 + 1)

    if(smallest_angle(ang_zero, phis_2[-1]) < smallest_angle(ang_zero, phis_2[0])):
        phis_2 = np.flip(phis_2, axis = 0)

    n_2 = int(analysis_params_general.num_outside_angle+1)
    phis_2 = phis_2[n_2:]

    y_ticks_2 = np.linspace(0, 1 + analysis_params_general.num_outside_angle*phi_unit_2/phi_range_2, angle_sel_num + analysis_params_general.num_outside_angle + 1)

    # if(is_plot):
    #   plot_angles(ang_zero, ang_one2r, phis_2)
    
    ### combining phis
    phis = np.concatenate((phis_1,phis_2), axis = 0)
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