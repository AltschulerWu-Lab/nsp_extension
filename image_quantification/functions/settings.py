# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2020-03-27 15:06:33
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-09 01:24:53

import io, os, sys, types, pickle, datetime, time

global paths, analysis_params_general, matching_info

def get_unique_filename(prefix, filename):
	name = filename.split('.')[0]
	filetype = '.'+filename.split('.')[1]
	if(os.path.exists(os.path.join(prefix, name+filetype))):
		i = 1
		sup = f'_0{i}{filetype}'
		while(os.path.exists(os.path.join(prefix, name+sup))):
			i +=1
			sup = f'_0{i}{filetype}'
		name = name + sup
		return name
	else:
		return name+filetype

### class that store all the paths.
class Paths:
	def __init__(self, env, names):
		### unpack parameters
		fig_out_folder, data_out_folder, annot_name, channels_type = names

		### initialization
		image_folder = 'Images'
		roi_folder = 'ROIs'
		annot_folder = 'Annotations'

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
			self.data_prefix = '/wynton/home/awlab/lmorinishi/data'
			self.output_prefix = '/wynton/home/awlab/lmorinishi/output'
			self.log_prefix = '/wynton/home/awlab/lmorinishi/code/logs'
			self.code_path = '/wynton/home/awlab/lmorinishi/code/functions'
		
		elif(env == 'Windows'):
			self.data_prefix = 'W:\\2019_09_NSP_Extension\\results\\Fate_Switching_Experiments\\Gal80\\Data_Gal80'
			self.output_prefix = 'W:\\2019_09_NSP_Extension\\results\\Fate_Switching_Experiments\\Gal80\\Output_Gal80'
			self.log_prefix = 'W:\\2019_09_NSP_Extension\\code\\NSP_extension\\python\\python_cluster\\logs'
			self.code_path = 'W:\\2019_09_NSP_Extension\\code\\NSP_extension\\python\\python_cluster\\functions'

		elif(env == 'Github'):
			# example image on github.
			self.data_prefix = os.path.join(os.path.dirname(os.getcwd()), 'data_example')
			self.output_prefix = os.path.join(os.path.dirname(os.getcwd()), 'output_example')
			self.log_prefix = os.path.join(os.path.dirname(os.getcwd()), 'logs')
			self.code_path = os.path.join(os.path.dirname(os.getcwd()), 'functions')

		### paths
		now = datetime.datetime.now()
		date_info=f'{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}'
		annot_name_sub = annot_name.split('.')[0]

		log_name = f'{annot_name_sub}_s{input_list[4]}c{input_list[5]}_{channels_type}_log_v{date_info}.txt'
		log_name = get_unique_filename(self.log_prefix, log_name)
		cat_name = annot_name_sub.split('_')[0]
		time_name = annot_name_sub.split('_')[1][:2]

		self.log_path = os.path.join(self.log_prefix, log_name)
		self.image_path = os.path.join(self.data_prefix, image_folder)
		self.roi_path = os.path.join(self.data_prefix, roi_folder)
		self.annot_path = os.path.join(self.data_prefix, annot_folder, annot_name)
		self.fig_out_prefix = os.path.join(self.output_prefix, fig_out_folder)
		self.data_out_prefix = os.path.join(self.output_prefix, data_out_folder)
		self.annot_name = annot_name

### class that stores indexing and color coding set-ups that are universal.
class MatchingInfo:
	def __init__(self, channels_type):
		# type of channels info.
		# R3R4: calculate density map of R3 and R4s
		# checking: checking raw images.
		self.channels_type = channels_type
		
		# dic{target array index : target ID}
		self.index_to_target_id = {
			0:0, 
			1:2, 
			2:3, 
			3:4, 
			4:5, 
			5:7
		} 
		
		# dic{target ID : target array index}
		self.target_id_to_index = {
			0:0, 
			2:1, 
			3:2, 
			4:3, 
			5:4, 
			7:5
		} 

		# dic{target ID : color code}
		self.color_code = {
			1: '#00FFFF', 
			2: '#1FF509', 
			3: '#FF0000', 
			4: '#CFCF1C', 
			5: '#FF00FF', 
			6: '#FFAE01', 
			7: '#FF7C80', 
			0: '#FFFFFF'
		}

		# name of channels for each channels_type
		if(channels_type == 'R3R4'):
			self.channel_mapping = {
				'RFP':0, 
				'GFP':1, 
				0:'RFP', 
				1:'GFP', 
			} 
			self.channel_cmap = {
				0:'Reds', 
				1:'Greens', 
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
		self.num_angle_section = int(input_list[5])
		self.num_outside_angle = int(input_list[6])
		self.num_x_section = int(input_list[7])
		self.z_offset = int(input_list[8])
		self.radius_expanse_ratio = 3

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
# num_angle_section = input_list[5]
# num_outside_angle = input_list[6]
# num_x_section = input_list[7]
# z_offset = input_list[8]
# channels_type = input[9]
# example of an input: Github, Figure_Output, Data_Output, Ctrl_26hrs_Gal80_s1r1_example.csv, 1, 10, 5, 10, 10, R3R4
input_str = input()
input_list = input_str.split(', ')
print(input_str)

### get paths class
names = [input_list[1], input_list[2], input_list[3], input_list[9]]
paths = Paths(input_list[0], names)

### get indexing class
matching_info = MatchingInfo(input_list[9])

### get analysis parameters
analysis_params_general = GeneralParams(input_list)