# -*- coding: utf-8 -*-
# @Author: sf942274
# @Date:   2019-07-28 16:51:03
# @Last Modified by:   sf942274
# @Last Modified time: 2019-11-14 08:25:09

import io, os, sys, types, datetime, pickle, warnings

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import numpy as np
from numpy.linalg import eig, inv

import math

from scipy import interpolate, spatial, stats

import seaborn as sns

import skimage.io as skiIo
from skimage import exposure, img_as_float

from sklearn import linear_model, metrics

from statannot import add_stat_annotation

import Data_quantification_function_helper as my_help
import Data_quantification_function_intensity_calculation as my_int
import Data_quantification_function_parse_bundle as my_pb
import Data_quantification_function_plotting as my_plot

def save_data_file(data, nameparams, outputDir):
    categoryID, sampleID, regionID = nameparams
    now = datetime.datetime.now()
    date_info = str(now.year)+str(now.month)+str(now.day)
    outputname = categoryID + '_sample' + str(sampleID) + '_region' + str(regionID) + '_v' + date_info + '.pickle'
    
    my_help.check_dir(outputDir)
    outputname = os.path.join(outputDir, outputname)
    pickle_out = open(outputname,"wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def dataframe_add_column(df, column_list):
    new_col = []
    for col in column_list:
        if(col not in df.columns):
            new_col.append(col)
    if(len(new_col) > 0):
        df = df.reindex( columns = df.columns.tolist() + new_col )
    return df

def p_value_calculation(df, hue_name, value_name, pair_list):
    df_groups = df.groupby(hue_name)
    for pair in pair_list:
        a = df_groups.get_group(pair[0]).loc[:,value_name].values
        b = df_groups.get_group(pair[1]).loc[:,value_name].values
        t_mwu, p_mwu = stats.mannwhitneyu(a,b)
        t_t, p_studentt = stats.ttest_ind(a,b)
        print(f'{pair}: Mann Whitney U test: {p_mwu}; Student T Test: {p_studentt}')

def plot_bar_plots(y, y_error, name, title, plot_options):
    title, ylim, figsize, tick_font_size, title_font_size = plot_options
    
    x_pos = np.arange(len(name))
    
    fig, ax = plt.subplots(figsize = figsize)
    ax.bar(x_pos, y, yerr=y_error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(name, fontsize = tick_font_size)
    ax.set_title('Relative length', fontsize = title_font_size)
    ax.yaxis.grid(True)
    ax.set_ylim(ylim)
    ax.grid(False)

def get_angle_unit():
    dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
    dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
    dT2T5 = dT3T3_ = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
    dT0T3 = dT0T3_ = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5

    ## Angles (in radius)
    aT2T0T5 = math.radians(80)
    aT0T5T4 = math.radians(100)
    aT3T0T3_ = math.acos((dT0T3 ** 2 + dT0T3_ ** 2 - dT3T3_ ** 2)/(2*dT0T3*dT0T3_))
    phi_unit = aT3T0T3_/2
    return phi_unit

def getTargetGridPolar_summary(pos_t3, pos_t7):
    dT0T2 = dT0T5 = dT2T4 = dT4T5 = 1
    dT0T4 = dT2T3 = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT4T5*dT0T5*math.cos(math.radians(100)))**0.5
    dT2T5 = dT3T3_ = (dT0T5 ** 2 + dT4T5 ** 2 -2*dT0T2*dT0T5*math.cos(math.radians(80)))**0.5
    dT0T3 = dT0T3_ = ((dT2T5/2) ** 2 + (dT2T3*1.5) ** 2) ** 0.5
    
    aT0T2 = math.radians(80)/2
    aT0T5 = - math.radians(80)/2
    aT0T3 = math.acos((dT0T3 ** 2 + dT0T3_ ** 2 - dT3T3_ ** 2)/(2*dT0T3*dT0T3_))/2
    aT0T3_ = - aT0T3
    aT0T4 = 0
    
    ## normalized axis
    dT0T4n = 1
    dT0T2n = dT0T2*dT0T4n/dT0T4
    dT0T3n = dT0T3*dT0T4n/dT0T4
    
    ## points
    T0 = np.array((0,0))
    T2 = np.array((aT0T2, dT0T2n))
    T3 = np.array((aT0T3, pos_t3))
    T4 = np.array((aT0T4, dT0T4n))
    T5 = np.array((aT0T5, dT0T2n))
    T3_ = np.array((aT0T3_, pos_t7))
    
    targetGridPolar = np.stack((T0, T2, T3, T4, T5, T3_), axis = 0)
    
    return targetGridPolar

def plotSummaryMatrix(matrix, cmap, vmax, tickParams, figureParams):
    xTick_ori, yTick_ori, tickTypeX, tickTypeY, tickArg2_X, tickArg2_Y = tickParams
    ylabel, xlabel, title = labelParams
    figsize, labelsize, titlesize = sizeParams
    xTicks = my_plot.getTickList(tickTypeX, xTick_ori, tickArg2_X)
    yTicks = my_plot.getTickList(tickTypeY, yTick_ori, tickArg2_Y)
    
    fig = plt.figure(figsize = figsize)
    ax2 = plt.subplot(111)
    if(vmax):
        sns.heatmap(matrix, cmap = cmap, yticklabels = yTicks, xticklabels = xTicks, vmin = 0, vmax = vmax)
    else:
        sns.heatmap(matrix, cmap = cmap, yticklabels = yTicks, xticklabels = xTicks)
    ax2.set_ylabel(ylabel, fontsize=labelsize)
    ax2.set_xlabel(xlabel, fontsize=labelsize)
    ax2.set_title(title, fontsize=titlesize)
    ax2.invert_yaxis()

def getPolarPlotValues(analysisParams, channelNo, matrix, cmap, targetIndexMatch_rev, radiusExpanseRatio, pos_t3, pos_t7, z_min, z_max):
    targetGridPolar = getTargetGridPolar_summary(pos_t3, pos_t7)
    targetGridPos = my_plot.getTargetGrid()
    CutOffPoint = 1 * radiusExpanseRatio
    bundleParams = targetGridPos[targetIndexMatch_rev[7]], targetGridPos[targetIndexMatch_rev[3]], CutOffPoint, targetGridPos[targetIndexMatch_rev[0]] 

    rs, phis = my_plot.getSliceParams_forFigure(analysisParams, bundleParams)
    thetav, rv = np.meshgrid(phis, rs)

    z = matrix.transpose()

    if(z_min == -1):
        z_min = z.min()
    if(z_max == -1):
        z_max = z.max()

    levels = MaxNLocator(nbins=15).tick_values(z_min, z_max)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    return thetav, rv, z, norm, targetGridPolar

def plot_summary_polar(matrix, channelNo, calculation_params, matching_info, figure_params):
#     vmax_set = 1
    analysisParams, radiusExpanseRatio, pos_t3, pos_t7 = calculation_params
    figsize, colormap, z_min, z_max = figure_params
    
    targetIndexMatch, ColorCode, channel_mapping, channel_cmap, targetIndexMatch_rev = matching_info
    
#     colormap = plt.get_cmap(channel_cmap[channelNo])
    # colormap = plt.get_cmap('gray')
    
    thetav, rv, z1, norm1, targetGridPolar = getPolarPlotValues(analysisParams, channelNo, matrix, colormap, targetIndexMatch_rev, radiusExpanseRatio, pos_t3, pos_t7, z_min, z_max)
    
    fig = plt.figure(figsize = figsize)
#     fig = plt.figure()
    ax2 = fig.add_subplot(111, polar = True)

    ## plot value
    sc = ax2.pcolormesh(thetav, rv, z1, cmap=colormap, norm=norm1)

    ## plot heel position
    for i in [2,3,5]:
        ax2.plot(targetGridPolar[i,0], targetGridPolar[i,1], 'o', color = ColorCode[targetIndexMatch[i]], markersize = 20, mew = 3, mfc = 'none')
    
    ## plot angle reference
    ax2.plot([0, targetGridPolar[targetIndexMatch_rev[3],0]], [0, targetGridPolar[targetIndexMatch_rev[3],1]], '--', color = '0.5')
    ax2.plot([0, targetGridPolar[targetIndexMatch_rev[7],0]], [0, targetGridPolar[targetIndexMatch_rev[7],1]], '--', color = '0.5')

    ## set polar to pie
    ax2.set_thetamin(-40)
    ax2.set_thetamax(40)
    
    ax2.set_rlim(0, 3)
    # ax.set_xticklabels({'fontsize':20})
    ax2.tick_params(axis = 'x', labelsize = 30)
    ax2.tick_params(axis = 'y', labelsize = 30)

    #### color bar for polar plot
    vmin,vmax = sc.get_clim()
#     vmax = 0.2
    cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)   #-- Defining a normalised scale
    ax5 = fig.add_axes([0.8, 0.2, 0.02, 0.6])       #-- Creating a new axes at the right side
    cb1 = matplotlib.colorbar.ColorbarBase(ax5, norm=cNorm, cmap=colormap, )    #-- Plotting the colormap in the created axes
    cb1.ax.tick_params(labelsize=20)
    fig.subplots_adjust(left=0.0,right=0.95)
    
    return fig

def process_sum_df(sum_df, annots_df, outputData):
    angle_unit = get_angle_unit()

    ### calculate angles
    sum_df.loc[:,'angle_max'] = np.mean([sum_df.loc[:,'angle_max1'].values, sum_df.loc[:,'angle_max2'].values], axis = 0)
    sum_df.loc[:,'angle_avg'] = np.mean([sum_df.loc[:,'angle_avg1'].values, sum_df.loc[:,'angle_avg2'].values], axis = 0)
    sum_df.loc[:,'angle_max_abs'] = np.abs(sum_df.loc[:,'angle_max'].values)
    sum_df.loc[:,'angle_avg_abs'] = np.abs(sum_df.loc[:,'angle_avg'].values)
    sum_df.loc[:,'angle_radian_max'] = sum_df.loc[:,'angle_max1'].values*angle_unit
    sum_df.loc[:,'angle_radian_avg'] = sum_df.loc[:,'angle_avg1'].values*angle_unit
    sum_df.loc[:,'angle_radian_max_abs'] = np.abs(sum_df.loc[:,'angle_radian_max'].values)
    sum_df.loc[:,'angle_radian_avg_abs'] = np.abs(sum_df.loc[:,'angle_radian_avg'].values)

    ### correct for heel position of R3 and R4s

    sum_df = dataframe_add_column(sum_df, ['length_max_fromheel', 'length_avg_fromheel', 'length_max_fromT3', 'length_avg_fromT3', 'pos_t3', 'pos_t7', 'heel_pos_type', 'bundle_no_total'])
    annots_df_group = annots_df.groupby(['CategoryID', 'SampleID', 'RegionID'])
    sum_df_group = sum_df.groupby(['CategoryID', 'SampleID', 'RegionID'])

    i_bundle_no_total = -1
    for iData in outputData.keys():
        category = outputData[iData]['categoryID']
        sampleID = outputData[iData]['sampleID']
        regionID = outputData[iData]['regionID']
        rel_pos_list = outputData[iData]['relativePositions']
        print("===")
        print(category, sampleID, regionID)
        
        sum_df_current = sum_df_group.get_group((category, sampleID, regionID))
        annots_df_current = annots_df_group.get_group((category, sampleID, regionID))
        annots_df_current.loc[:,'Bundle_No'] = annots_df_current.loc[:,'Bundle_No'].values.astype(int)
        annots_df_current.set_index('Bundle_No', inplace = True)
        
        for ind_annot, bundle_no in enumerate(annots_df_current.index):
            sum_df.loc[ii,'bundle_no_total'] = i_bundle_no_total
            # print(ind_annot, bundle_no)
            r3_heel = rel_pos_list[ind_annot,7]
            r4_heel = rel_pos_list[ind_annot,8]
            t3_pos = rel_pos_list[ind_annot,2]
            t7_pos = rel_pos_list[ind_annot,5]
            inds_sum =  sum_df_current.index[(sum_df_current['bundle_no'] == bundle_no)]
            # print(inds_sum)
            
            if(len(inds_sum) > 0):
                r_types = sum_df_current.loc[inds_sum, 'type_Rcell']
                if(len(r_types.values.flatten()) > 2):
                    print('Error! multiple incidents of same bundle!!')
                else:
                    if(sum_df_current.loc[inds_sum,['type_bundle']].values.flatten()[0] == 'R3R4'): # R3R4 case
                        r_types = sum_df_current.loc[inds_sum,['type_Rcell']]
                        for ii in r_types.index:
                            r_type = r_types.loc[ii, 'type_Rcell']
                            sum_df.loc[ii,'pos_t3'] = t3_pos
                            sum_df.loc[ii,'pos_t7'] = t7_pos
                            
                            if(r_type == 3):
                                sum_df.loc[ii,'heel_pos_type'] = 3
                                sum_df.loc[ii,['length_max_fromheel','length_avg_fromheel']] = sum_df.loc[ii,['length_max', 'length_avg']].values - r3_heel
                                sum_df.loc[ii,['length_max_fromT3', 'length_avg_fromT3']] = t3_pos - sum_df.loc[ii,['length_max', 'length_avg']].values
                            elif(r_type == 4):
                                sum_df.loc[ii,'heel_pos_type'] = 4
                                sum_df.loc[ii,['length_max_fromheel','length_avg_fromheel']] = sum_df.loc[ii,['length_max', 'length_avg']].values - r4_heel
                                sum_df.loc[ii,['length_max_fromT3', 'length_avg_fromT3']] = t7_pos - sum_df.loc[ii,['length_max', 'length_avg']].values
                            else:
                                print('error! non 3 nor 4!')
                    else:
                        angle1 = sum_df.loc[r_types.index[0], 'angle_avg']
                        angle2 = sum_df.loc[r_types.index[1], 'angle_avg']
                        if(angle1 > angle2):
                            iR3 = r_types.index[0]
                            iR4 = r_types.index[1]
                        else:
                            iR3 = r_types.index[1]
                            iR4 = r_types.index[0]
                        sum_df.loc[iR3,'heel_pos_type'] = 3
                        sum_df.loc[iR4,'heel_pos_type'] = 4

                        sum_df.loc[iR3,['length_max_fromheel','length_avg_fromheel']] = sum_df.loc[iR3,['length_max', 'length_avg']].values - r3_heel
                        sum_df.loc[iR4,['length_max_fromheel','length_avg_fromheel']] = sum_df.loc[iR4,['length_max', 'length_avg']].values - r4_heel

                        sum_df.loc[iR3,['length_max_fromT3', 'length_avg_fromT3']] = t3_pos - sum_df.loc[iR3,['length_max', 'length_avg']].values
                        sum_df.loc[iR4,['length_max_fromT3', 'length_avg_fromT3']] = t7_pos - sum_df.loc[iR4,['length_max', 'length_avg']].values

                        sum_df.loc[iR3,'pos_t3'] = t3_pos
                        sum_df.loc[iR3,'pos_t7'] = t7_pos
                        sum_df.loc[iR4,'pos_t3'] = t3_pos
                        sum_df.loc[iR4,'pos_t7'] = t7_pos

    sum_df_groups = sum_df.groupby(['type_Rcell', 'type_bundle'])
    sum_df.loc[sum_df_groups.get_group((3, 'R3R3')).index,'type_plot'] = 'R3/R3'
    sum_df.loc[sum_df_groups.get_group((4, 'R4R4')).index,'type_plot'] = 'R4/R4'
    sum_df.loc[sum_df_groups.get_group((3, 'R3R4')).index,'type_plot'] = 'R3'
    sum_df.loc[sum_df_groups.get_group((4, 'R3R4')).index,'type_plot'] = 'R4'

    return sum_df

def get_summary_matrices(sum_df, annots_df, outputData):
    annots_df_group = annots_df.groupby(['CategoryID', 'SampleID', 'RegionID'])
    sum_df_group = sum_df.groupby(['CategoryID', 'SampleID', 'RegionID'])

    num_channels = outputData[0]['intensityMatrix'].shape[1]
    matrixY = outputData[0]['intensityMatrix'].shape[2]
    matrixX = outputData[0]['intensityMatrix'].shape[3]
    matrixZ = outputData[0]['intensityMatrix'].shape[4]
    sum_df_group = sum_df.groupby(['CategoryID', 'SampleID', 'RegionID'])
    matrices = {
        'Fz':{
            'R3R4':np.zeros((1, num_channels, matrixY, matrixX, matrixZ)) - 100,
            'R3R3':np.zeros((1, num_channels, matrixY, matrixX, matrixZ)) - 100,
            'R4R4':np.zeros((1, num_channels, matrixY, matrixX, matrixZ)) - 100
            },
        'N':{
            'R3R4':np.zeros((1, num_channels, matrixY, matrixX, matrixZ)) - 100,
            'R4R4':np.zeros((1, num_channels, matrixY, matrixX, matrixZ)) - 100,
            'R3R3':np.zeros((1, num_channels, matrixY, matrixX, matrixZ)) - 100
            }
    }

    for iData in outputData.keys():
    # iData = 0
        category = outputData[iData]['categoryID']
        sampleID = outputData[iData]['sampleID']
        regionID = outputData[iData]['regionID']
        rel_pos_list = outputData[iData]['relativePositions']
        print("===")
        print(category, sampleID, regionID)

        sum_df_current = sum_df_group.get_group((category, sampleID, regionID))
        annots_df_current = annots_df_group.get_group((category, sampleID, regionID))
        annots_df_current.loc[:,'Bundle_No'] = annots_df_current.loc[:,'Bundle_No'].values.astype(int)
        annots_df_current.set_index('Bundle_No', inplace = True)
        annots_df_current.sort_index(axis = 0, inplace = True)
        print(outputData[iData]['intensityMatrix'].shape[0], len(annots_df_current.index))

        for ind, bundle_no in enumerate(annots_df_current.index):
    #         print(ind, bundle_no)
            sumdf_inds =  sum_df_current.index[(sum_df_current.loc[:,'bundle_no'] == bundle_no)]
            matrix = np.expand_dims(outputData[iData]['intensityMatrix'][ind,:,:,:,:], axis = 0)
            if(len(sumdf_inds) > 0):
                type_bundle = sum_df_current.loc[sumdf_inds,'type_bundle'].values[0]
                matrices[category][type_bundle]= np.concatenate((matrices[category][type_bundle], matrix), axis = 0)
                if(category == 'Fz'):
    #                 Fz_matrix[type_bundle] = np.concatenate((Fz_matrix[type_bundle], matrix), axis = 0)
                    if(type_bundle == 'R3R3'):
                        print('R3R3:', ind, bundle_no)
                    elif(type_bundle == 'R3R4'):
                        print('R3R4:', ind, bundle_no)
                elif(category == 'N'):
    #                 N_matrix[type_bundle] = np.concatenate((N_matrix[type_bundle], matrix), axis = 0)
                    if(type_bundle == 'R4R4'):
                        print('R4R4:', ind, bundle_no)
                    elif(type_bundle == 'R3R4'):
                            print('R3R4:', ind, bundle_no)
    for category in matrices.keys():
        for type_bundle in matrices[category].keys():
            matrices[category][type_bundle] = matrices[category][type_bundle][1:,:,:,:,:]
    # for keys in Fz_matrix:
    #     Fz_matrix[keys] = Fz_matrix[keys][1:,:,:,:,:]
    #     N_matrix[keys] = N_matrix[keys][1:,:,:,:,:]

    return matrices