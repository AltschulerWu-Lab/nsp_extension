# -*- coding: utf-8 -*-
# @Author: sf942274
# @Date:   2019-07-15 04:32:53
# @Last Modified by:   sf942274
# @Last Modified time: 2019-07-30 02:32:15

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
    input_list = input().split(', ')

    ### directory
    dataPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Data_Analysis/Fate_Switching_experiments/Data_Gal80'
    figOutPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Data_Analysis/Fate_Switching_experiments/Output_Gal80/FigureOutput_trial'
    dataOutPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Data_Analysis/Fate_Switching_experiments/Output_Gal80/DataOutput_trial'
    logPrefix = '/awlab/projects/2015_07_Neural_Superposition/Projects/Weiyue/Code/python_cluster/log_txts'

    # file folders
    imageFolder = 'Images'
    ROIFolder = 'ROIs'
    annotFolder = 'Annotations'
    annotName = input_list[0]

    now = datetime.datetime.now()
    date_info = str(now.year)+str(now.month)+str(now.day)+str(now.hour)
    nn = annotName.split('.')[0]
    logName = f'{nn}_s{input_list[1]}c{input_list[2]}_log_v{date_info}.txt'
    # logName = annotName.split('.')[0] +'_log' + '_v' + date_info + '.txt'

    # paths
    imagePath = os.path.join(dataPrefix, imageFolder)
    ROIPath = os.path.join(dataPrefix, ROIFolder)
    annotPath = os.path.join(dataPrefix, annotFolder, annotName)
    logPath = os.path.join(logPrefix, logName)

    paths = [imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix]

    ### analysis parameters
    slicetype = int(input_list[1])
    centertype = int(input_list[2])

    radiusExpanseRatio = [2.5, 3]
    num_angleSection = int(input_list[3])
    num_outsideAngle = int(input_list[4])
    num_Xsection = int(input_list[5])
    z_offset = int(input_list[6])
    
    scale_factor = float(input_list[7])

    analysisParams_general = (num_angleSection, num_outsideAngle, num_Xsection, z_offset, radiusExpanseRatio, slicetype, centertype)

    ### color codes
    targetIndexMatch = {0:0, 1:2, 2:3, 3:4, 4:5, 5:7}
    targetIndexMatch_rev = {0:0, 2:1, 3:2, 4:3, 5:4, 7:5}
    ColorCode = {1:'#00FFFF', 2:'#1FF509', 3: '#FF0000', 4: '#CFCF1C', 5: '#FF00FF', 6: '#FFAE01', 7:'#983535', 0:'#FFFFFF'}
    channel_mapping = {'RFP':0, 'GFP':1, 'R3_1':2, 'R4_1':3, 'R3_2':4, 'R4_2':5, 'R3_3': 6, 0:'RFP', 1:'GFP', 2:'R3_1', 3:'R4_1', 4:'R3_2', 5:'R4_2', 6:'R3_3'}
    channel_cmap = {0:'Reds', 1: 'Greens', 2:'Reds', 3: 'Greens', 4:'Reds', 5: 'Greens', 6:'Reds'}
    matching_info = (targetIndexMatch, ColorCode, channel_mapping, channel_cmap, targetIndexMatch_rev)
    
    ### print
    print("=====" +annotName + " Analysis Start! =====")
    print_to_log(logPath, "=====" + annotName + " Analysis Start! =====")

    return paths, analysisParams_general, matching_info, scale_factor

def print_to_log(logPath, info):
    logFile = open(logPath, "a+")
    logFile.write(info + '\n')
    logFile.close()

def import_data(paths):
    imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix = paths

    summary_df = pd.read_csv(annotPath)
    image_list = summary_df.loc[:,'Image_Name'].unique()
    ROI_list = summary_df.loc[:,'ROI_Name'].unique()

    i_image = 0

    imageName = image_list[i_image]
    ROIName = ROI_list[i_image]
    ROI_df = pd.read_csv(os.path.join(ROIPath, ROIName))
    ROI_df.rename(columns = {' ':'No'}, inplace = True)
    annot_df = summary_df.groupby(['Image_Name']).get_group(image_list[i_image]).reset_index(drop = True)

    image = img_as_float(skiIo.imread(os.path.join(imagePath, imageName)))
    image_shape = (image.shape[0], image.shape[1], image.shape[2])
    M2P_ratio = (summary_df.iloc[0]['imgX_pixel']/summary_df.iloc[0]['imgX_um'], summary_df.iloc[0]['imgY_pixel']/summary_df.iloc[0]['imgY_um'])

    image_info = [imageName, image_shape, M2P_ratio]

    return ROI_df, annot_df, image, image_info

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

def analyze_image(bundles_df, annot_bundles_df, paths, analysisParams_general, matching_info, image_norm, image_name):
    ### parameters
    imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix = paths
    num_angleSection, num_outsideAngle, num_Xsection, z_offset, radiusExpanseRatio, slicetype, centertype = analysisParams_general
    targetIndexMatch, ColorCode, channel_mapping, channel_cmap, targetIndexMatch_rev = matching_info

    ### initialization
    print('-----' + image_name + '------')
    print_to_log(logPath, '-----' + image_name + '------')
    matrixY = num_angleSection + 2*num_outsideAngle + 1
    matrixX = num_Xsection + 1
    matrixZ = z_offset * 2 + 1
    num_norm_channels = image_norm.shape[-1]

    IntensityMatrix = np.zeros((len(annot_bundles_df.index), num_norm_channels, matrixY, matrixX, matrixZ))
    IntensityMatrix = IntensityMatrix - 100
    #IntensityMatrix.shape = ind, channelNo, matrixX, matrixY, matrixZ
    params = [];
    rel_points = np.zeros((len(annot_bundles_df.index), 9))

    ### thresholds
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

    ### process
    for ind, bundle_No in enumerate(annot_bundles_df.index):
        R_Z = int(annot_bundles_df.loc[bundle_No,'coord_Z_R' + str(3)]) - 1
        print("Bundle No: " + str(bundle_No))
        print_to_log(logPath, "Bundle No: " + str(bundle_No))

        ### targets info
        indTs, coordTs = my_help.getTargetCoords(bundle_No, bundles_df, targetIndexMatch)
        coord_Center = my_help.getBundleCenter(bundle_No, bundles_df)
        coordR4s = my_help.getRxCoords(bundle_No, bundles_df, indTs, 4)
        coordR3s = my_help.getRxCoords(bundle_No, bundles_df, indTs, 3)
        coordRs = np.concatenate((coordR4s, coordR3s))

        ### slice info
        SliceZeroPoint = coordTs[targetIndexMatch_rev[7],:] # T3'
        SliceOnePoint = coordTs[targetIndexMatch_rev[3],:] # T3

        LengthOnePoint = coordTs[targetIndexMatch_rev[4],:]

        CenterPoints = [coordTs[0,:], coord_Center[0,:]]

        Rcell_nums = [4,4]

        printingParams = [False, False]

        ### get slicing params and calculate matrix
        centertype = centertype
        slicetype = slicetype
        analysis_params = [analysisParams_general[0], analysisParams_general[1], analysisParams_general[2], analysisParams_general[3], analysisParams_general[4][centertype]]
        bundleParams = [bundle_No, indTs, coordTs, coord_Center, SliceZeroPoint, SliceOnePoint, LengthOnePoint, CenterPoints[centertype], Rcell_nums[centertype]]
        if(slicetype == 0):
            pp_i, rel_points_i = my_int.getSliceParams_v1(analysis_params, bundles_df, bundleParams, printingParams, matching_info[4])
        elif(slicetype == 1):
            pp_i, rel_points_i = my_int.getSliceParams_v3(analysis_params, bundles_df, bundleParams, printingParams, matching_info[4])
        params.append(pp_i)
        rel_points[ind, :] = rel_points_i

        # calculate matrix
        time_start = time.time()
        for channelNo in range(num_norm_channels):
            print_to_log(logPath, "Channle No: " + str(channelNo))
            # IntensityMatrix[ind, channelNo,:,:,:] = my_int.getIntensityMatrix_new(pp_i, image_norm, channel_mapping[channelNo], channel_mapping)
            IntensityMatrix[ind, channelNo,:,:,:] = np.random.randn(IntensityMatrix[ind, channelNo,:,:,:].shape[0], IntensityMatrix[ind, channelNo,:,:,:].shape[1], IntensityMatrix[ind, channelNo,:,:,:].shape[2])
        time_end = time.time()
        time_dur = time_end - time_start
        print_to_log(logPath, "total time: " + str(time_dur))

    return IntensityMatrix, params, rel_points, thr_otsu, thr_li, thr_isodata

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

                figname = f'{categoryID}_s{sampleID}r{regionID}_Bundle_No_{bundle_No}_{thrFunction}'
                figParams = [centertype, pp_i, figOutPrefix, img_name, figname, slicetype, radiusExpanseRatio[centertype]]
                plotOptions = [True, True, True, True, False, thrs, thrFunction, num_norm_channels] ### isPlotLine, isLabelOff, isSave, isTrueXTick, isOriTick, thrFunction            
                fig = my_plot.plotBundleVsMatrix_all(bundle_No, bundles_df, image_norm, matrix, figParams, tickParams, plotOptions, matching_info)
                plt.close(fig)

            ## polar plot
            figParams = pp_i, figOutPrefix, img_name, radiusExpanseRatio[centertype], centertype, slicetype
            plotOptions = [True, True] # isLabelOff, isSave
            for channelNo in range(num_norm_channels):
                analysis_params = [analysisParams_general[0], analysisParams_general[1], analysisParams_general[2], analysisParams_general[3], analysisParams_general[4][centertype]]
                fig = my_plot.plotPolar(bundle_No, bundles_df, image_norm, analysis_params, channelNo, matrix, figParams, plotOptions, matching_info, rel_points_i)
                plt.close(fig)

        else:
            print("error! No intensity matrix calculated!")


def save_results(annot_bundles_df, IntensityMatrix, params, rel_points, paths, slicetype, centertype):
    imagePath, ROIPath, annotPath, logPath, figOutPrefix, dataOutPrefix = paths

    categoryID = annot_bundles_df.iloc[0]['CategoryID']
    sampleID = annot_bundles_df.iloc[0]['SampleID']
    regionID = annot_bundles_df.iloc[0]['RegionID']

    outputData = {
        'categoryID' : categoryID,
        'sampleID' : sampleID,
        'regionID':regionID,
        'sliceType':slicetype,
        'centerType':centertype,
        'intensityMatrix':IntensityMatrix,
        'parameter':params,
        'relativePositions':rel_points
    }

    now = datetime.datetime.now()
    date_info = str(now.year)+str(now.month)+str(now.day)
    outputname = f'{categoryID}_sample{sampleID}_region{regionID}_slice{slicetype}_center{centertype}_v{date_info}.pickle'

    outputDir = os.path.join(dataOutPrefix)
    my_help.check_dir(outputDir)
    outputDir = os.path.join(outputDir,categoryID)
    my_help.check_dir(outputDir)
    outputname = os.path.join(outputDir, outputname)
    pickle_out = open(outputname,"wb")
    pickle.dump(outputData, pickle_out)
    pickle_out.close()


def main():
    
    paths, analysisParams_general, matching_info, scale_factor = set_parameters()

    logPath = paths[3]

    ROI_df, annot_df, image, image_info = import_data(paths)
    print("Data import finished!")
    print_to_log(logPath, "Data import finished!")
    
    bundles_df, annot_bundles_df = process_annotation(ROI_df, annot_df, image_info[2])
    print("annot_bundles_df done!")
    print_to_log(logPath, "annot_bundles_df done!")
    
    time_start = time.time()
    image_norm = process_image(image, image_info[1], logPath, matching_info[3], scale_factor)
    time_end = time.time()
    time_dur = time_end - time_start
    print("image processed! total time: ", time_dur)
    print_to_log(logPath, "image processed! total time: " + str(time_dur))
    
    IntensityMatrix, params, rel_points, thr_otsu, thr_li, thr_isodata = analyze_image(bundles_df, annot_bundles_df, paths, analysisParams_general, matching_info, image_norm, image_info[0])
    print("image analyzed!")
    print_to_log(logPath, "image analyzed!")

    save_results(annot_bundles_df,IntensityMatrix, params, rel_points, paths, analysisParams_general[5], analysisParams_general[6])
    print("data saved!")
    print_to_log(logPath, "data saved!")
    
    print_results(bundles_df, annot_bundles_df, IntensityMatrix, params, rel_points, paths, analysisParams_general, matching_info, image_norm, image_info[0], thr_otsu, thr_li, thr_isodata)
    print("image results generated!")
    print_to_log(logPath, "image results generated!")

    

if __name__ == "__main__":
    main()

