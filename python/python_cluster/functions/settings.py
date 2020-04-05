# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2020-03-27 15:06:33
# @Last Modified by:   sf942274
# @Last Modified time: 2020-04-05 02:36:14

import io, os, sys, types, pickle, datetime, time

global paths, analysis_params_general, matching_info

### class that store all the paths.
class Paths:
	def __init__(self, env, names):
		### unpack parameters
		fig_out_folder, data_out_folder, annot_name, channels_type = names

		### initialization
		image_folder = 'Images'
		roi_folder = 'ROIs'
		annot_folder = 'Annotations'
		now = datetime.datetime.now()
		date_info = str(now.year)+str(now.month)+str(now.day)+str(now.hour)
		annot_name_sub = annot_name.split('.')[0]

		### paths depends on where the code is running
		if(env == 'Mac'):
			self.data_prefix = '/Users/lily/Lily/Academic/AW_Lab/data/Gal80_data'
			self.output_prefix = '/Users/lily/Lily/Academic/AW_Lab/data/Gal80_output'
			self.log_prefix = '/Users/lily/Lily/Academic/AW_Lab/data/logs'
			self.code_path = '/Users/lily/Lily/Academic/AW_Lab/code/NSP_extension/python/python_cluster/functions'

		elif(env == 'Mac_HD'):
			self.data_prefix = '/Volumes/WJI_Lab/NSP_analysis/Data_Gal80'
			self.output_prefix = '/Volumes/WJI_Lab/NSP_analysis/Output'
			self.log_prefix = '/Users/lily/Lily/Academic/AW_Lab/data/logs'
			self.code_path = '/Users/lily/Lily/Academic/AW_Lab/code/NSP_extension/python/python_cluster/functions'
		
		elif(env == 'Euclid'):
			# euclid with cluster
			self.data_prefix = '/awlab/projects/2019_09_NSP_Extension/results/Fate_Switching_Experiments/Gal80/Data_Gal80'
			self.output_prefix = '/awlab/projects/2019_09_NSP_Extension/results/Fate_Switching_Experiments/Gal80/Output_Gal80/'
			self.log_prefix = '/awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/logs'
			self.code_path = '/awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/functions'
		
		elif(env == 'Wynton'): 
			# home directory of Wynton cluster
			self.data_prefix = '/wynton/home/awlab/wji/data'
			self.output_prefix = '/wynton/home/awlab/wji/output'
			self.log_prefix = '/wynton/home/awlab/wji/code/logs'
			self.code_path = '/wynton/home/awlab/wji/code/functions'
		
		elif(env == 'Windows'):
			self.data_prefix = 'W:\\2019_09_NSP_Extension\\results\\Fate_Switching_Experiments\\Gal80_temporal\\Data'
			self.output_prefix = 'W:\\2019_09_NSP_Extension\\results\\Fate_Switching_Experiments\\Gal80_temporal\\Output'
			self.log_prefix = 'W:\\2019_09_NSP_Extension\\code\\NSP_codes\\python_cluster\\logs'
			self.code_path = 'W:\\2019_09_NSP_Extension\\code\\NSP_codes\\python_cluster\\functions'

		### paths
		log_name = f'{annot_name_sub}_s{input_list[4]}c{input_list[5]}_{channels_type}_log_v{date_info}.txt'
		cat_name = annot_name_sub.split('_')[0]
		time_name = annot_name_sub.split('_')[1][:2]

		self.log_path = os.path.join(self.log_prefix, log_name)
		self.image_path = os.path.join(self.data_prefix, image_folder)
		self.roi_path = os.path.join(self.data_prefix, roi_folder)
		self.annot_path = os.path.join(self.data_prefix, annot_folder, cat_name, time_name, annot_name)
		self.fig_out_prefix = os.path.join(self.output_prefix, fig_out_folder)
		self.data_out_prefix = os.path.join(self.output_prefix, data_out_folder)
		self.annot_name = annot_name

### class that stores indexing and color coding set-ups that are universal.
class MatchingInfo:
	def __init__(self, channels_type):
		self.channels_type = channels_type
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
		if(channels_type == 'R3R4'):
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
				6:'R3_3',
			} 
			self.channel_cmap = {
				0:'Reds', 
				1:'Greens', 
				2:'Reds', 
				3:'Greens', 
				4:'Reds', 
				5:'Greens', 
				6:'Reds',
			}
		elif(channels_type == 'FasII'):
			self.channel_mapping = {
				'RFP':0,
				'GFP':1,
				'FasII':2,
				'R3_FasII':3,
				'R4_FasII':4,
				0:'RFP',
				1:'GFP',
				2:'FasII',
				3:'R3_FasII',
				4:'R4_FasII'
			}
			self.channel_cmap = {
				0:'Reds', 
				1:'Greens', 
				2:'Blues', 
				3:'Reds', 
				4:'Greens', 
			}
		elif(channels_type == 'checking'):
			self.channel_mapping = {
				0:'RFP',
				1:'GFP',
				2:'24b10',
				3:'FasII'
			}
			self.channel_cmap = {
				0:'Reds', 
				1:'Greens', 
				2:'Greys', 
				3:'Blues', 
			}


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
		self.radius_expanse_ratio = [3, 3.8]

	def add_col_params(self, dic):
		if not hasattr(self, 'col_params'):
			self.col_params = {}
			if dic:
				for key in dic:
					self.col_params[key] = dic[key]
			else:
				self.col_params = {}
		else:
			if dic:
				for key in dic:
					self.col_params[key] = dic[key]
			else:
				self.col_params = {}


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
# channels_type = input[11]
# example of an input: Windows, Data_Output, Figure_Output, Fz_26hrs_Gal80_s5r1_annotation.csv, 0, 1, 10, 5, 10, 10, 1, FasII
input_str = input()
input_list = input_str.split(', ')
print(input_str)

### get paths class
names = [input_list[1], input_list[2], input_list[3], input_list[11]]
paths = Paths(input_list[0], names)

### get indexing class
matching_info = MatchingInfo(input_list[11])

### get analysis parameters
analysis_params_general = GeneralParams(input_list)