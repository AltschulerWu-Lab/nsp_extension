import io, os, sys, types, pickle

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
from skimage import exposure, img_as_float, filters

from sklearn import linear_model
from sklearn import metrics


""" ============== path settings =============="""

global isMac
isMac = False

if(isMac):
	dataPrefix = '/Users/lily/Lily/Academic/AW_Lab/data/fate_switching_gfp_rfp_old/Data'
	figOutPrefix = '/Users/lily/Lily/Academic/AW_Lab/data/fate_switching_gfp_rfp_old/Output/FigureOutput'
	dataOutPrefix = '/Users/lily/Lily/Academic/AW_Lab/data/fate_switching_gfp_rfp_old/Output/DataOutput'
else:
	sys.path.insert(0, 'Z:\\lab\\Projects\\NSP\\NSP_Code\\python\\Quantification_fromPositionOnly\\Functions')
	dataPrefix = 'Z:\\lab\\Projects\\NSP\\Data_Analysis\\Experiments\\Fate_Switching\\GFP_RFP\\Summaries\\Data'
	figOutPrefix = 'Z:\\lab\\Projects\\NSP\\Data_Analysis\\Experiments\\Fate_Switching\\GFP_RFP\\Summaries\\Outputs_new\\FigureOutput_v0422'
	dataOutPrefix = 'Z:\\lab\\Projects\\NSP\\Data_Analysis\\Experiments\\Fate_Switching\\GFP_RFP\\Summaries\\Outputs_new\\DataOutput_v0422'

### file parameters
# file folders
imageFolder = 'Images'
ROIFolder = 'ROIs'
summaryFolder = 'Annotations\\28hrs'
dataFolder = 'DataOutput'

# file name
summaryName = 'Control_s3r1_summary.csv'



""" ============== Golbal variables and parameters =============="""



### global variables
global ColorCode, targetIndexMatch, bins, channel_mapping
bins = 4096



### analysis parameters
radiusExpanseRatio = 1.5
num_angleSection = 20
num_outsideAngle = 20
num_Xsection = 20
z_offset = 25
analysisParams = [num_angleSection, num_outsideAngle, num_Xsection, z_offset, radiusExpanseRatio]



### import custom functions
import Data_quantification_function_helper as my_help
import Data_quantification_function_intensity_calculation as my_int
import Data_quantification_function_parse_bundle as my_pb
import Data_quantification_function_plotting as my_plot


"""============== main =============="""

sns.set_style("dark")

"""Load data"""
### load summaries
summary_df = pd.read_csv(os.path.join(dataPrefix, summaryFolder, summaryName))
image_list = summary_df.loc[:,'Image_Name'].unique()
ROI_list = summary_df.loc[:,'ROI_Name'].unique()
isExtendedTargetList = False


### Extended target list or not
if(isExtendedTargetList == True):
	targetIndexMatch = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:30, 8:20, 9:50, 10:40}
	targetIndexMatch_rev = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 30:7, 20:8, 50:9, 40:10}
	ColorCode = {0:'#FFFFFF', 1:'#52FEFE', 2:'#1FF509', 3: '#FF0000', 4: '#CFCF1C', 5: '#FF00FF', 6: '#FFAE01', 20:'#1FF509', 30:'#FF0000', 40:'#CFCF1C', 50:'#FF00FF'}
	channel_mapping = {'RFP':0, 'GFP':1, 'R3':2, 'R4':3, '24B10':4, 0:'RFP', 1:'GFP', 2:'R3', 3:'R4', 4:'24B10'}
	matching_info = (targetIndexMatch, ColorCode, channel_mapping, targetIndexMatch_rev)
else:
	targetIndexMatch = {0:0, 1:2, 2:3, 3:4, 4:5, 5:7}
	targetIndexMatch_rev = {0:0, 2:1, 3:2, 4:3, 5:4, 7:5}
	ColorCode = {1:'#00FFFF', 2:'#1FF509', 3: '#FF0000', 4: '#CFCF1C', 5: '#FF00FF', 6: '#FFAE01', 7:'#FF0000', 0:'#FFFFFF'}
	channel_mapping = {'RFP':0, 'GFP':1, 'R3':2, 'R4':3, 'FasII':4, '24B10':5, 0:'RFP', 1:'GFP', 2:'R3', 3:'R4', 4:'FasII', 5:'24B10'}
	matching_info = (targetIndexMatch, ColorCode, channel_mapping, targetIndexMatch_rev)

### load other data
i_image = 0
imageName = image_list[i_image]
ROIName = ROI_list[i_image]
print(imageName, ROIName)
image = img_as_float(skiIo.imread(os.path.join(dataPrefix, imageFolder, imageName)))
ROI_df = pd.read_csv(os.path.join(dataPrefix, ROIFolder, ROIName))
ROI_df.rename(columns = {' ':'No'}, inplace = True)
annot_df = summary_df.groupby(['Image_Name']).get_group(image_list[i_image]).reset_index(drop = True)
image_shape = (image.shape[0], image.shape[1], image.shape[2])
M2P_ratio = (summary_df.iloc[0]['imgX_pixel']/summary_df.iloc[0]['imgX_um'], summary_df.iloc[0]['imgY_pixel']/summary_df.iloc[0]['imgY_um'])
print("Data loaded!")

""" Process annotation info"""
bundles_df = my_pb.getBundlesInfo(ROI_df, annot_df, M2P_ratio[0], M2P_ratio[1], isExtendedTargetList)
annot_bundles_df = bundles_df.dropna(axis=0, how='any', inplace = False)
annot_bundles_df_good = my_pb.Good_QC_df(annot_bundles_df)
annot_bundles_df_bad = my_pb.Bad_QC_df(annot_bundles_df)

""" Process images """
### number of channels
nChannels = image.shape[3]
if(nChannels == 3):
	print("3 channels!")
	### Seperate channels
	GFP = image[:,:,:,1]
	RFP = image[:,:,:,0]
	Cy5 = image[:,:,:,2]

	### normalize GFP & RFP channel
	image_norm = np.empty(image_shape + (nChannels + 2,), dtype=GFP.dtype, order='C')
	
#     %time GFP_norm = exposure.equalize_adapthist(GFP)
	GFP_norm = exposure.rescale_intensity(GFP, in_range = 'image', out_range='dtype')
	
#     %time RFP_norm = exposure.equalize_adapthist(RFP)
	RFP_norm = exposure.rescale_intensity(RFP, in_range = 'image', out_range='dtype')
	
#     %time Cy5_norm = exposure.equalize_adapthist(GFP)
	Cy5_norm = exposure.rescale_intensity(Cy5, in_range = 'image', out_range='dtype')
	
	image_norm[:,:,:,0] = RFP_norm
	image_norm[:,:,:,1] = GFP_norm
	image_norm[:,:,:,4] = Cy5_norm

	R3 = RFP_norm - GFP_norm
	R3[R3<0] = 0
#     %time R3_norm = exposure.equalize_adapthist(R3)
	R3_norm = exposure.rescale_intensity(R3, in_range = 'image', out_range='dtype')
	image_norm[:,:,:,2] = R3_norm

	R4 = RFP_norm * GFP_norm
#     %time R4_norm = exposure.equalize_adapthist(R4)
	R4_norm = exposure.rescale_intensity(R4, in_range = 'image', out_range='dtype')
	image_norm[:,:,:,3] = R4_norm
	
elif(nChannels == 4):
	print("4 channels!")
	### Seperate channels
	GFP = image[:,:,:,1]
	RFP = image[:,:,:,0]
	Cy5 = image[:,:,:,2]
	DAPI = image[:,:,:,3]

	### normalize GFP & RFP channel
	image_norm = np.empty(image_shape + (nChannels + 2,), dtype=GFP.dtype, order='C')
	
#     %time GFP_norm = exposure.equalize_hist(GFP)
	GFP_norm = exposure.rescale_intensity(GFP, in_range = 'image', out_range='dtype')
	
#     %time RFP_norm = exposure.equalize_hist(RFP)
	RFP_norm = exposure.rescale_intensity(RFP, in_range = 'image', out_range='dtype')
	
	Cy5_norm = exposure.rescale_intensity(Cy5, in_range = 'image', out_range='dtype')
	DAPI_norm = exposure.rescale_intensity(DAPI, in_range = 'image', out_range='dtype')
	
	image_norm[:,:,:,0] = RFP_norm
	image_norm[:,:,:,1] = GFP_norm
	image_norm[:,:,:,4] = DAPI_norm
	image_norm[:,:,:,5] = Cy5_norm

	R3 = RFP_norm - GFP_norm
	R3[R3<0] = 0
	R3_norm = exposure.rescale_intensity(R3, in_range = 'image', out_range='dtype')
	image_norm[:,:,:,2] = R3_norm

	R4 = RFP_norm * GFP_norm
	R4_norm = exposure.rescale_intensity(R4, in_range = 'image', out_range='dtype')
	image_norm[:,:,:,3] = R4_norm


""" Plot individual bundles """


# In[ ]:


ind = 1
bundle_No = list(annot_bundles_df.index)[ind]
plotSettings = False, False, False, True #isPlotR3Line, isPlotR4Line, isPlotR4s, isLabelOff
my_plot.plotIndividualBundles(bundle_No, bundles_df, image_norm, M2P_ratio[0], M2P_ratio[1], plotSettings, matching_info)


# In[ ]:


""" get intensity matrix for good bundles"""


# In[ ]:


### initialization
print('-----' + image_list[i_image] + '------')
matrixX = num_angleSection + 2*num_outsideAngle + 1
matrixY = num_Xsection + 1
matrixZ = z_offset * 2 + 1
IntensityMatrix = np.zeros((4, len(annot_bundles_df.index), matrixX, matrixY, matrixZ))
params = [];


# In[ ]:


image_norm.shape


# In[ ]:


IntensityMatrix.shape


# In[ ]:


### thresholds
thr_otsu = np.zeros((4))
thr_li = np.zeros((4))
for channelNo in range(4):
	get_ipython().run_line_magic('time', 'thr_otsu[channelNo] = filters.threshold_otsu(image_norm[:,:,:,channelNo])')
#     %time thr_li[channelNo] = filters.threshold_li(image_norm[:,:,:,channelNo])


# In[ ]:





# In[ ]:


### parse bundles
# bundle_No = 76
# for ind in range(len(annot_bundles_df.index)):
ind = 0
bundle_No = list(annot_bundles_df.index)[ind]
R_Z = int(bundles_df.loc[bundle_No,'coord_Z_R' + str(3)]) - 1
print("Bundle No: ", bundle_No)
#     DV = bundles_df.loc[bundle_No, 'Orientation_DV']
#     AP = bundles_df.loc[bundle_No, 'Orientation_AP']

### targets info
indTs, coordTs = my_help.getTargetCoords(bundle_No, bundles_df, targetIndexMatch)
coordR4s = my_help.getRxCoords(bundle_No, bundles_df, indTs, 4)
coordR3s = my_help.getRxCoords(bundle_No, bundles_df, indTs, 3)
coordRs = np.concatenate((coordR4s, coordR3s))

### slice info
SliceZeroPoint = coordTs[targetIndexMatch_rev[7],:] # T3'
SliceOnePoint = coordTs[targetIndexMatch_rev[3],:] # T3

## slice radius info
CutOffPoints = []
CutOffPoints.append(coordR3s[0,:] + (coordTs[targetIndexMatch_rev[4],:] - coordTs[0,:]) * radiusExpanseRatio) # T4-T0
CutOffPoints.append(coordR4s[0,:] + (coordTs[targetIndexMatch_rev[4],:] - coordTs[0,:]) * radiusExpanseRatio) # T4-T0
CutOffPoints.append(coordTs[0,:] + (coordTs[targetIndexMatch_rev[4],:] - coordTs[0,:]) * radiusExpanseRatio) # T4-T0

CenterPoints = [coordR3s[0,:], coordR4s[0,:], coordTs[0,:]]

Rcell_nums = [3,4,4]

printingParams = [False, False]

## get slicing params
mind = 2 # use T0 as center
sliceTypeNo = 0 # first slicing method
bundleParams = [bundle_No, indTs, coordTs, SliceZeroPoint, SliceOnePoint, CutOffPoints[mind], CenterPoints[mind], Rcell_nums[mind]]
pp = my_int.getSliceParams_v1(analysisParams, bundles_df, bundleParams, printingParams)
params.append(pp)

### calculate matrix
for channelNo in range(4):
	print(channelNo)
	IntensityMatrix[channelNo,ind,:,:,:] = my_int.getIntensityMatrix_new(pp, image_norm, channel_mapping[channelNo], channel_mapping)


# In[ ]:


### plotting: heatmap
matrix = IntensityMatrix[:, ind, :,:,:]
plt.ioff()
img_name = image_list[i_image]
ori_X = np.round(np.linspace(0, radiusExpanseRatio, matrix.shape[2]), 2)
tickParams = [2, 1, ori_X, 21] ### tickTypeX, tickTypeY, tickArg2_X, tickArg2_Y
figParams = [mind, pp[0:7], figOutPrefix, img_name, sliceTypeNo, radiusExpanseRatio]
for thrFunction in [0,1]: # different thresholding methods
	if(thrFunction == 0):
		thrs = np.zeros((4))
	elif(thrFunction == 1):
		thrs = thr_otsu
	#     elif(thrFunction == 2):
	#         thrs = thr_li
plotOptions = [True, True, False, True, False, thrs, thrFunction] ### isPlotLine, isLabelOff, isSave, isTrueXTick, isOriTick, thrFunction            
fig = my_plot.plotBundleVsMatrix_all(bundle_No, bundles_df, image_norm, matrix, figParams, tickParams, plotOptions, matching_info)
# plt.close(fig)

### plotting: polar
figParams = pp[0:7], figOutPrefix, img_name, radiusExpanseRatio
plotOptions = [True, False] # isLabelOff, isSave
for channelNo in [0,1,2,3]:
# channelNo = 0
	fig = my_plot.plotPolar(bundle_No, bundles_df, image_norm, analysisParams, channelNo, matrix, figParams, plotOptions, matching_info)
#     plt.close(fig)


# In[ ]:


[1150.73498409  254.23638724]


# In[ ]:


params[0]


# In[ ]:


np.save('ctrl_matrix_bundleNo90.npy', matrix)


# In[ ]:


### Save matrixes and parameters


# In[ ]:


categoryID = annot_bundles_df.iloc[0]['CategoryID']
sampleID = annot_bundles_df.iloc[0]['SampleID']
regionID = annot_bundles_df.iloc[0]['RegionID']


# In[ ]:


outputData = {}
outputData['categoryID'] = categoryID
outputData['sampleID'] = sampleID
outputData['regionID'] = regionID
outputData['IntensityMatrix'] = IntensityMatrix
outputData['Parameter'] = params


# In[ ]:


import datetime


# In[ ]:


now = datetime.datetime.now()
date_info = str(now.year)+str(now.month)+str(now.day)
outputname = categoryID + '_sample' + str(sampleID) + '_region' + str(regionID) + '_v' + date_info + '.pickle'


# In[ ]:


outputDir = os.path.join(dataOutPrefix)
my_help.check_dir(outputDir)
outputDir = os.path.join(outputDir,categoryID)
my_help.check_dir(outputDir)
# outputname = categoryID + '_sample' + str(sampleID) + '_region' + str(regionID) +  + '.pickle'
outputname = os.path.join(outputDir, outputname)
pickle_out = open(outputname,"wb")
pickle.dump(outputData, pickle_out)
pickle_out.close()


# In[ ]:


IntensityMatrix.shape


# In[ ]:


np.nonzero(IntensityMatrix)


# In[ ]:





# In[ ]:


# ind = 1

# # mind = 2
# # sliceType = 1
# channelNo = 2
# # 
# channel = channel_mapping[channelNo]
# bundle_No = list(annot_bundles_df.index)[ind]

# # plt.ioff()
# img_name = image_list[i_image]
# figParams1 = [channel, mind, pp[sliceTypeNo][mind], figOutPrefix, img_name, sliceTypeNo, radiusExpanseRatio]
# figParams2 = [mind, pp[sliceTypeNo][mind], figOutPrefix, img_name, sliceTypeNo, radiusExpanseRatio]

# tickParams = [1, 2, 5, [-1, -0.5, 0, 0.5, 1]] ### tickTypeX, tickTypeY, tickArg2_X, tickArg2_Y
# plotOptions = [True, True, False, True, False] ### isPlotLine, isLabelOff, isSave, isTrueXTick, isMask
# matrix1 = IntensityMatrix[sliceTypeNo, channelNo, mind, ind, :,:,:]
# matrix2 = IntensityMatrix[sliceTypeNo, :, mind, ind, :,:,:]
# # fig1 = my_plot.plotBundleVsMatrix(bundle_No, bundles_df, image_norm, matrix1, figParams1, tickParams, plotOptions, matching_info)
# fig2 = plotBundleVsMatrix_all(bundle_No, bundles_df, image_norm, matrix2, figParams2, tickParams, plotOptions, matching_info)
# # plt.close(fig)

