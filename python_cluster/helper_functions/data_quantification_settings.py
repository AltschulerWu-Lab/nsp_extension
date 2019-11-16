import io, os, sys, types, pickle, datetime, time

### get input
input_list = input().split(', ')

### directory
data_prefix = '/awlab/projects/2019_09_NSP_Extension/figure/analysis/Fate_Switching_experiments/Data_Gal80'
fig_out_prefix = '/awlab/projects/2019_09_NSP_Extension/figure/analysis/Fate_Switching_experiments/Output_Gal80/FigureOutput_v1017'
data_out_prefix = '/awlab/projects/2019_09_NSP_Extension/figure/analysis/Fate_Switching_experiments/Output_Gal80/DataOutput_v1017'
log_prefix = '/awlab/projects/2019_09_NSP_Extension/code/python_cluster/logs'

# file folders
image_folder = 'Images'
roi_folder = 'ROIs'
annot_folder = 'Annotations'
annot_name = input_list[0]

now = datetime.datetime.now()
date_info = str(now.year)+str(now.month)+str(now.day)+str(now.hour)
nn = annot_name.split('.')[0]
logName = f'{nn}_s{input_list[1]}c{input_list[2]}_log_v{date_info}.txt'
# logName = annot_name.split('.')[0] +'_log' + '_v' + date_info + '.txt'

# paths
image_path = os.path.join(data_prefix, image_folder)
roi_path = os.path.join(data_prefix, roi_folder)
annot_path = os.path.join(data_prefix, annot_folder, annot_name)
log_path = os.path.join(log_prefix, logName)

### analysis parameters
slice_type = int(input_list[1])
center_type = int(input_list[2])

radius_expanse_ratio = [2.5, 3]
num_angle_section = int(input_list[3])
num_outside_angle = int(input_list[4])
num_x_section = int(input_list[5])
z_offset = int(input_list[6])

### color codes
index_to_target_id = {
	0:0, 
	1:2, 
	2:3, 
	3:4, 
	4:5, 
	5:7
} # index: target ID 
target_id_to_index = {
	0:0, 
	2:1, 
	3:2, 
	4:3, 
	5:4, 
	7:5
} # target ID : index
color_code = {
	1:'#00FFFF', 
	2:'#1FF509', 
	3: '#FF0000', 
	4: '#CFCF1C', 
	5: '#FF00FF', 
	6: '#FFAE01', 
	7:'#983535', 
	0:'#FFFFFF'
} # color code for each target ID
channel_mapping = {
	'RFP':0, 
	'GFP':1, 
	'R3_1':2, 
	'R4_1':3, 
	'R3_2':4, 
	'R4_2':5, 
	'R3_3': 6, 
	0:'RFP', 
	1:'GFP', 
	2:'R3_1', 
	3:'R4_1', 
	4:'R3_2', 
	5:'R4_2', 
	6:'R3_3'
} # mapping between image dimention and channels
channel_cmap = {
	0:'Reds', 
	1:'Greens', 
	2:'Reds', 
	3:'Greens', 
	4:'Reds', 
	5:'Greens', 
	6:'Reds'
} #  

class Paths:
	image_path = image_path
	roi_path = roi_path
	annot_path = annot_path
	log_path = log_path
	fig_out_prefix = fig_out_prefix
	data_out_prefix = data_out_prefix

class GeneralParams:
	num_angle_section = num_angle_section
	num_outside_angle = num_outside_angle
	num_x_section = num_x_section
	z_offset = z_offset
	radius_expanse_ratio = radius_expanse_ratio
	slice_type = slice_type
	center_type = center_type

class MatchingInfo:
	index_to_target_id = index_to_target_id
	color_code = color_code
	channel_mapping = channel_mapping
	channel_cmap = channel_cmap
	target_id_to_index = target_id_to_index

global paths, analysis_params_general, matching_info, scale_factor
paths = Paths()
analysis_params_general = GeneralParams()
matching_info = MatchingInfo()
scale_factor = float(input_list[7])

