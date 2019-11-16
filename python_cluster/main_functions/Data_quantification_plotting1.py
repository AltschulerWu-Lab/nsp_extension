# -*- coding: utf-8 -*-
# @Author: sf942274
# @Date:   2019-07-15 04:32:53
# @Last Modified by:   lily
# @Last Modified time: 2019-08-13 12:23:40

# make it snake_case!!!!!!!

import io, os, sys, types, pickle, datetime, time

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
from skimage import exposure, img_as_float, filters, morphology, transform

from sklearn import linear_model, metrics

sys.path.insert(0, '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Code/python_cluster/helper_functions')

import Data_quantification_function_helper as my_help
import Data_quantification_function_intensity_calculation as my_int
import Data_quantification_function_parse_bundle as my_pb
import Data_quantification_function_plotting as my_plot

def set_parameters():
    ### get input
    # input_list = input().split(', ')

    ### directory
    dataPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Data_Analysis/Fate_Switching_experiments/Data_Gal80'
    figOutPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Data_Analysis/Fate_Switching_experiments/Output_Gal80/FigureOutput_v0812'
    dataOutPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Data_Analysis/Fate_Switching_experiments/Output_Gal80/DataOutput_v0812'
    logPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Code/python_cluster/log_txts'

    # file folders
    image_folder = 'Images'
    ROI_folder = 'ROIs'
    annotation_folder = 'Annotations'

    logName = '20190813_plotting_1.txt'

    # paths
    imagePath = os.path.join(dataPrefix, image_folder)
    ROIPath = os.path.join(dataPrefix, ROI_folder)
    annotPath = os.path.join(dataPrefix, annotation_folder)
    logPath = os.path.join(logPrefix, logName)

    paths = [imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix]

    ### analysis parameters
    slice_type = 0
    center_type = 1
    radius_expanse_ratio = [2.5, 3]
    num_angle_section = 24
    num_outside_angle = 18
    num_x_section = 40
    z_offset = 20
    
    analysisParams_general = (num_angle_section, num_outside_angle, num_x_section, z_offset, radius_expanse_ratio, slice_type, center_type)

    ### color codes
    targetIndexMatch = {0:0, 1:2, 2:3, 3:4, 4:5, 5:7}
    targetIndexMatch_rev = {0:0, 2:1, 3:2, 4:3, 5:4, 7:5}
    ColorCode = {1:'#00FFFF', 2:'#1FF509', 3: '#FF0000', 4: '#CFCF1C', 5: '#FF00FF', 6: '#FFAE01', 7:'#983535', 0:'#FFFFFF'}
    channel_mapping = {'RFP':0, 'GFP':1, 'R3_1':2, 'R4_1':3, 'R3_2':4, 'R4_2':5, 'R3_3': 6, 0:'RFP', 1:'GFP', 2:'R3_1', 3:'R4_1', 4:'R3_2', 5:'R4_2', 6:'R3_3'}
    channel_cmap = {0:'Reds', 1: 'Greens', 2:'Reds', 3: 'Greens', 4:'Reds', 5: 'Greens', 6:'Reds'}
    matching_info = (targetIndexMatch, ColorCode, channel_mapping, channel_cmap, targetIndexMatch_rev)
    
    ### print
    print("=====" + " Printing Start! =====")
    print_to_log(logPath, "=====" + " Printing Start! =====")

    return paths, analysisParams_general, matching_info

def print_to_log(logPath, info):
    logFile = open(logPath, "a+")
    logFile.write(info + '\n')
    logFile.close()

def import_data(paths):
    imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix = paths

    folders, files = my_help.parseFolderInfo(annotPath)
    for filename in files:
        print(filename)
        if('.csv' in filename):
            df_temp = pd.read_csv(os.path.join(annotPath, filename))
            if(files.index(filename) == 0):
                annots_df = df_temp
            else:
                df_temp = pd.read_csv(os.path.join(annotPath, filename))
                annots_df = annots_df.append(df_temp, ignore_index=True, sort=True)

    outputData = {}
    filepaths = my_help.getFilePaths(os.path.join(dataOutPrefix, 'Selected', '1'))
    for i in range(len(filepaths)):
        print(f'====={i}=====')
        pickle_in = open(filepaths[i],"rb")
        outputData[i] = pickle.load(pickle_in)
        category = outputData[i]['categoryID']
        sampleID = outputData[i]['sampleID']
        regionID = outputData[i]['regionID']
        print(category, sampleID, regionID)

    return annots_df, outputData

def process_annotation(ROI_df, annot_df, M2P_ratio):
    isExtendedTargetList = False
    annotation_type = annot_df.loc[0,'Annotation_type']
    
    if(annotation_type == 1):
        bundles_df = my_pb.getBundlesInfo_v1(ROI_df, annot_df, M2P_ratio[0], M2P_ratio[1], isExtendedTargetList)
    elif(annotation_type == 2):
        bundles_df = my_pb.getBundlesInfo_v2(ROI_df, annot_df, M2P_ratio[0], M2P_ratio[1], isExtendedTargetList)
    
    annot_bundles_df = bundles_df.dropna(axis=0, how='any', inplace = False)

    annot_bundles_df.sort_index(inplace = True)
    
    return bundles_df, annot_bundles_df

def process_image(image, image_shape, logPath, channel_cmap, scale_factor):
    ### number of channels
    nChannels = image.shape[3]
    num_norm_channels = len(channel_cmap.keys()) # number of channels of normalized image

    print("4 channels!")
    print_to_log(logPath, "4 channels!")

    ### normalize channels
    image_norm = np.empty(image_shape + (num_norm_channels,), dtype=image[:,:,:,1].dtype, order='C')
    thr = np.zeros((2))
    
    # RFP_norm
    image_norm[:,:,:,0] = exposure.rescale_intensity(image[:,:,:,0], in_range = 'image', out_range='dtype')
    # GFP_norm
    image_norm[:,:,:,1] = exposure.rescale_intensity(image[:,:,:,1], in_range = 'image', out_range='dtype')    

    del image
    
    print("gfp threshold!")
    print_to_log(logPath, "gfp threshold!")
    thr[0] = filters.threshold_isodata(image_norm[:,:,:,1])
    thr[1] = filters.threshold_mean(image_norm[:,:,:,1])

    print("histogram matching!")
    print_to_log(logPath, "histogram matching!")
    gfp = transform.match_histograms(image_norm[:,:,:,1], image_norm[:,:,:,0])
    
    print("R3/R4 v1")
    print_to_log(logPath, "R3/R4 v1")
    R3 = image_norm[:,:,:,0] - gfp
    R3[R3<0] = 0
    image_norm[:,:,:,2] = exposure.rescale_intensity(R3, in_range = 'image', out_range='dtype')
    R4 = image_norm[:,:,:,0] * gfp
    image_norm[:,:,:,3] = exposure.rescale_intensity(R4, in_range = 'image', out_range='dtype')
    
    print("R3/R4 v2")
    print_to_log(logPath, "R3/R4 v2")
    gfp_thr = morphology.binary_opening((image_norm[:,:,:,1]>thr[0])*1)
    image_norm[:,:,:,4] = exposure.rescale_intensity(image_norm[:,:,:,0] * (1-gfp_thr), in_range = 'image', out_range='dtype')
    image_norm[:,:,:,5] = exposure.rescale_intensity(morphology.closing(image_norm[:,:,:,1]*((image_norm[:,:,:,1]>((thr[0] + thr[1])/2))*1)))

    print("R3 v3")
    print_to_log(logPath, "R3 v3")
    R3 = image_norm[:,:,:,0] - gfp*scale_factor
    R3[R3<0] = 0
    image_norm[:,:,:,6] = exposure.rescale_intensity(R3, in_range = 'image', out_range='dtype')

    del R3, R4, gfp, gfp_thr

    return image_norm

def calculate_thresholds(image_norm, logPath):
    num_norm_channels = image_norm.shape[-1]

    thr_otsu = np.zeros((num_norm_channels))
    thr_li = np.zeros((num_norm_channels))
    thr_isodata = np.zeros((num_norm_channels))
    time_start = time.time()
    for channelNo in range(num_norm_channels):
        thr_otsu[channelNo] = filters.threshold_otsu(image_norm[:,:,:,channelNo])
        thr_li[channelNo] = filters.threshold_li(image_norm[:,:,:,channelNo])
        thr_isodata[channelNo] = filters.threshold_isodata(image_norm[:,:,:,channelNo])
    time_end = time.time()
    time_dur = time_end - time_start
    print_to_log(logPath, "total time: " + str(time_dur))
    return thr_otsu, thr_li, thr_isodata

def print_results(bundles_df, annot_bundles_df, IntensityMatrix, params, rel_points, paths, analysisParams_general, matching_info, image_norm, img_name, thr_otsu, thr_li, thr_isodata):
    ### parameters
    imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix = paths
    num_angleSection, num_outsideAngle, num_Xsection, z_offset, radiusExpanseRatio, slicetype, centertype = analysisParams_general
    targetIndexMatch, ColorCode, channel_mapping, channel_cmap, targetIndexMatch_rev = matching_info

    num_norm_channels = image_norm.shape[-1]
    
    for ind, bundle_No in enumerate(annot_bundles_df.index):
        R_Z = int(bundles_df.loc[bundle_No,'coord_Z_R' + str(3)]) - 1
        print("Bundle No: ", bundle_No)
        print_to_log(logPath, "Bundle No: " + str(bundle_No))

        categoryID = annot_bundles_df.iloc[0]['CategoryID']
        sampleID = annot_bundles_df.iloc[0]['SampleID']
        regionID = annot_bundles_df.iloc[0]['RegionID']

        ### targets info
        indTs, coordTs = my_help.getTargetCoords(bundle_No, bundles_df, targetIndexMatch)
        coord_Center = my_help.getBundleCenter(bundle_No, bundles_df)
        coordR4s = my_help.getRxCoords(bundle_No, bundles_df, indTs, 4)
        coordR3s = my_help.getRxCoords(bundle_No, bundles_df, indTs, 3)
        coordRs = np.concatenate((coordR4s, coordR3s))

        ### parameters
        pp_i = params[ind]
        rel_points_i = rel_points[ind, :]

        matrix = my_help.delete_zero_columns(IntensityMatrix[ind, :, :, :, :], -100, 3)
        if(len(matrix.flatten()) > 0):

            ## heat map
            plt.ioff()
            ori_X = np.round(np.linspace(0, radiusExpanseRatio[centertype], matrix.shape[2]), 2)
            tickParams = [2, 1, ori_X, 21] ### tickTypeX, tickTypeY, tickArg2_X, tickArg2_Y
            for thrFunction in [0, 1, 2, 3]: # different thresholding methods
                if(thrFunction == 0):
                    thrs = np.zeros((num_norm_channels))
                elif(thrFunction == 1):
                    thrs = thr_otsu
                elif(thrFunction == 2):
                    thrs = thr_li
                elif(thrFunction == 3):
                    thrs = thr_isodata

                figname = f'{categoryID}_s{sampleID}r{regionID}_Bundle_No_{bundle_No}_{thrFunction}.png'
                figParams = [centertype, pp_i, figOutPrefix, img_name, figname, slicetype, radiusExpanseRatio[centertype]]
                plotOptions = [True, True, True, True, False, thrs, thrFunction, num_norm_channels] ### isPlotLine, isLabelOff, isSave, isTrueXTick, isOriTick, thrFunction            
                fig = my_plot.plotBundleVsMatrix_all(bundle_No, bundles_df, image_norm, matrix, figParams, tickParams, plotOptions, matching_info)
                plt.close(fig)

            ## polar plot
            # figParams = pp_i, figOutPrefix, img_name, radiusExpanseRatio[centertype], centertype, slicetype
            # plotOptions = [True, True] # isLabelOff, isSave
            # for channelNo in range(num_norm_channels):
            #     analysis_params = [analysisParams_general[0], analysisParams_general[1], analysisParams_general[2], analysisParams_general[3], analysisParams_general[4][centertype]]
            #     fig = my_plot.plotPolar(bundle_No, bundles_df, image_norm, analysis_params, channelNo, matrix, figParams, plotOptions, matching_info, rel_points_i)
            #     plt.close(fig)

        else:
            print("error! No intensity matrix calculated!")


def print_data(annots_df, outputData, paths, analysisParams_general, matching_info):
    imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix = paths
    num_angle_section, num_outside_angle, num_x_section, z_offset, radius_expanse_ratio, slice_type, center_type = analysisParams_general
    targetIndexMatch, ColorCode, channel_mapping, channel_cmap, targetIndexMatch_rev = matching_info

    annots_df_group = annots_df.groupby(['CategoryID', 'SampleID', 'RegionID'])
    for iData in outputData.keys():

        category = outputData[iData]['categoryID']
        sampleID = outputData[iData]['sampleID']
        regionID = outputData[iData]['regionID']

        print_text = f'====== categoryID: {category}, sampleID: {sampleID}, regionID: {regionID} ======'
        print(print_text)
        print_to_log(logPath, print_text)

        intensity_matrix = outputData[iData]['intensityMatrix']
        params = outputData[iData]['parameter']
        rel_points = outputData[iData]['relativePositions']

        annots_df_current = annots_df_group.get_group((category, sampleID, regionID))
        annots_df_current.loc[:,'Bundle_No'] = annots_df_current.loc[:,'Bundle_No'].values.astype(int)
        annots_df_current.reset_index(inplace = True)
        image_list = annots_df_current.loc[:,'Image_Name'].unique()
        ROI_list = annots_df_current.loc[:,'ROI_Name'].unique()

        i_image = 0
        imageName = image_list[i_image]
        ROI_name = ROI_list[i_image]
        ROI_df = pd.read_csv(os.path.join(ROIPath, ROI_name))
        ROI_df.rename(columns = {' ':'No'}, inplace = True)

        image = img_as_float(skiIo.imread(os.path.join(imagePath, imageName)))
        image_shape = (image.shape[0], image.shape[1], image.shape[2])
        M2P_ratio = (annots_df_current.iloc[0]['imgX_pixel']/annots_df_current.iloc[0]['imgX_um'], annots_df_current.iloc[0]['imgY_pixel']/annots_df_current.iloc[0]['imgY_um'])

        bundles_df, annot_bundles_df = process_annotation(ROI_df, annots_df_current, M2P_ratio)
        image_norm = process_image(image, image_shape, logPath, channel_cmap, 1)

        thr_otsu, thr_li, thr_isodata = calculate_thresholds(image_norm, logPath)

        print_results(bundles_df, annot_bundles_df, intensity_matrix, params, rel_points, paths, analysisParams_general, matching_info, image_norm, imageName, thr_otsu, thr_li, thr_isodata)

def main():
    
    paths, analysisParams_general, matching_info = set_parameters()

    logPath = paths[3]

    annots_df, outputData = import_data(paths)
    print("Data import finished!")
    print_to_log(logPath, "Data import finished!")
    
    print_data(annots_df, outputData, paths, analysisParams_general, matching_info)    
    print("image results generated!")
    print_to_log(logPath, "image results generated!")

if __name__ == "__main__":
    main()

