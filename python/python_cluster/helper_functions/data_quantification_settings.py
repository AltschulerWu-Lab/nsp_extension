# -*- coding: utf-8 -*-
# Author: lily
# Date:   2020-02-11 04:40:57
# Last Modified by:   sf942274
# Last Modified time: 2020-02-14 17:26:29
import io, os, sys, types, pickle, datetime, time

global paths, analysis_params_general, matching_info

### class that store all the paths.
class Paths:
	def __init__(self, env, names):
		### unpack parameters
		fig_out_folder, data_out_folder, annot_name = names

		### initialization
		image_folder = 'Images'
		roi_folder = 'ROIs'
		annot_folder = 'Annotations'
		now = datetime.datetime.now()
		date_info = str(now.year)+str(now.month)+str(now.day)+str(now.hour)
		annot_name_sub = annot_name.split('.')[0]

		### paths depends on where the code is running
		if(env == 'Mac'):
			data_prefix = '/Users/lily/Lily/Academic/AW_Lab/data/Gal80_data'
			output_prefix = '/Users/lily/Lily/Academic/AW_Lab/data/Gal80_output'
			log_prefix = '/Users/lily/Lily/Academic/AW_Lab/code/python_cluster/logs'
			self.code_path = '/Users/lily/Lily/Academic/AW_Lab/code/python_cluster/helper_functions'
		
		elif(env == 'Euclid'):
			# euclid with cluster
			data_prefix = '/awlab/projects/2019_09_NSP_Extension/figure/analysis/Fate_Switching_experiments/Data_Gal80'
			output_prefix = '/awlab/projects/2019_09_NSP_Extension/figure/analysis/Fate_Switching_experiments/Output_Gal80/'
			log_prefix = '/awlab/projects/2019_09_NSP_Extension/code/python_cluster/logs'
			self.code_path = '/Users/lily/Lily/Academic/AW_Lab/code/python_cluster/helper_functions'
		
		elif(env == 'Wynton'): 
			# home directory of Wynton cluster
			data_prefix = '/wynton/home/awlab/wji/data'
			output_prefix = '/wynton/home/awlab/wji/output'
			log_prefix = '/wynton/home/awlab/wji/code/logs'
			self.code_path = '/wynton/home/awlab/wji/code/helper_functions'
		
		elif(env == 'Windows'):
			data_prefix = 'W:\\2019_09_NSP_Extension\\results\\Fate_Switching_Experiments\\Gal80_temporal\\Data'
			output_prefix = 'W:\\2019_09_NSP_Extension\\results\\Fate_Switching_Experiments\\Gal80_temporal\\Output'
			log_prefix = 'W:\\2019_09_NSP_Extension\\code\\NSP_codes\\python_cluster\\logs'
			self.code_path = 'W:\\2019_09_NSP_Extension\\code\\NSP_codes\\python_cluster\\helper_functions'

		### paths
		log_name = f'{annot_name_sub}_s{input_list[1]}c{input_list[2]}_log_v{date_info}.txt'
		
		self.log_path = os.path.join(log_prefix, log_name)
		self.image_path = os.path.join(data_prefix, image_folder)
		self.roi_path = os.path.join(data_prefix, roi_folder)
		self.annot_path = os.path.join(data_prefix, annot_folder, annot_name)
		self.fig_out_prefix = os.path.join(output_prefix, fig_out_folder)
		self.data_out_prefix = os.path.join(output_prefix, data_out_folder)
		self.annot_name = annot_name

### class that stores indexing and color coding set-ups that are universal.
class MatchingInfo:
	def __init__(self):
		self.index_to_target_id = {
			0:0, 
			1:2, 
			2:3, 
			3:4, 
			4:5, 
			5:7
		} # dic{index : target ID}
		self.target_id_to_index = {
			0:0, 
			2:1, 
			3:2, 
			4:3, 
			5:4, 
			7:5
		} # dic{target ID : index}
		self.color_code = {
			1: '#00FFFF', 
			2: '#1FF509', 
			3: '#FF0000', 
			4: '#CFCF1C', 
			5: '#FF00FF', 
			6: '#FFAE01', 
			7: '#FF7C80', 
			0: '#FFFFFF'
		} # dic{color code for each R and target ID}
		self.channel_mapping = {
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
		} # mapping between image matrix dimention and channels
		self.channel_cmap = {
			0:'Reds', 
			1:'Greens', 
			2:'Reds', 
			3:'Greens', 
			4:'Reds', 
			5:'Greens', 
			6:'Reds'
		} # cmap used for plotting for each channel 

### class that stores parameters for analysis
class GeneralParams:
	def __init__(self, input_list):
		self.slice_type = int(input_list[4])
		self.center_type = int(input_list[5])
		self.num_angle_section = int(input_list[6])
		self.num_outside_angle = int(input_list[7])
		self.num_x_section = int(input_list[8])
		self.z_offset = int(input_list[9])
		self.scale_factor = float(input_list[10])
		self.radius_expanse_ratio = [2.5, 3]

### get input
# inputs:
# env = input_list[0]
# fig_out_folder = input_list[1]
# data_out_folder = input_list[2]
# annot_name = input_list[3]
# slice_type = input_list[4]
# center_type = input_list[5]
# num_angle_section = input_list[6]
# num_outside_angle = input_list[7]
# num_x_section = input_list[8]
# z_offset = input_list[9]
# scale_factor = int(input_list[10])
# example of an input: Windows, Data_Output, Figure_Output, Fz_26hrs_s5r1_annotation.csv, 0, 1, 10, 5, 10, 10, 1
input_str = input()
input_list = input_str.split(', ')
print(input_str)

### get paths class
names = [input_list[1], input_list[2], input_list[3]]
paths = Paths(input_list[0], names)

### get indexing class
matching_info = MatchingInfo()

### get analysis parameters
analysis_params_general = GeneralParams(input_list)