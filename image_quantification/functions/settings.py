# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2020-03-27 15:06:33
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-10 15:19:26

import io, os, sys, types, pickle, datetime, time

global paths, analysis_params_general, matching_info

def get_unique_filename(prefix, filename):
	"""
	Function: check if a filename exist within a folder (prefix). If yes, return a new name.
	Inputs:
	- prefix: path. Folder which the fild will be in.
	- filename: string. Name of the file.
	Output: string. Name of the file.
	"""
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

def check_dir(path):
	"""
	Function: check if a path exist. If not, make it.
	Input: path
	"""
	if not os.path.exists(path):
		if os.path.exists(os.path.dirname(path)):
			os.makedirs(path)
		else:
			check_dir(os.path.dirname(path))

### class that store all the paths.
class Paths:
	def __init__(self, annot_name):
		### paths to be specified. Can directly change from here.
		self.data_folder_path = os.path.join(os.path.dirname(os.getcwd()), 'data_example')
		self.output_folder_path = os.path.join(os.path.dirname(os.getcwd()), 'output_example')
		self.log_folder_path = os.path.join(os.path.dirname(os.getcwd()), 'logs')
		self.code_path = os.path.join(os.path.dirname(os.getcwd()), 'functions')

		### internal folder structure
		image_folder = 'Images'
		roi_folder = 'ROIs'
		annot_folder = 'Annotations'
		fig_out_folder = 'Figure_Output'
		data_out_folder = 'Data_Output'
		
		### check if log folder exist. If not, make one.
		check_dir(self.log_folder_path)

		### other paths
		now = datetime.datetime.now()
		date_info=f'{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}'
		annot_name_sub = annot_name.split('.')[0]

		log_name = f'{annot_name_sub}_log_v{date_info}.txt'
		log_name = get_unique_filename(self.log_folder_path, log_name)

		self.log_path = os.path.join(self.log_folder_path, log_name)
		self.image_folder_path = os.path.join(self.data_folder_path, image_folder)
		self.roi_folder_path = os.path.join(self.data_folder_path, roi_folder)
		self.annot_path = os.path.join(self.data_folder_path, annot_folder, annot_name)
		self.fig_out_folder_path = os.path.join(self.output_folder_path, fig_out_folder)
		self.data_out_folder_path = os.path.join(self.output_folder_path, data_out_folder)
		self.annot_name = annot_name

### class that stores indexing and color coding set-ups that are universal.
class MatchingInfo:
	def __init__(self):		
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

		# name of channels for processed images
		self.channel_mapping = {
			'RFP':0, 
			'GFP':1, 
			0:'RFP', 
			1:'GFP', 
		} 

		# color-map for each channel.
		self.channel_cmap = {
			0:'Reds', 
			1:'Greens', 
		}

### class that stores parameters for analysis
class GeneralParams:
	def __init__(self, input_list):
		self.slice_type = int(input_list[1]) # whether or not to force center-T4 = 0 in defining relative angles.
		self.num_angle_section = int(input_list[2]) # number of angle sections between T3 and T3'.
		self.num_outside_angle = int(input_list[3]) # number of angle sections to expand in calculation outside of angle(T3-c-T3').
		self.num_x_section = int(input_list[4]) # number of length sections
		self.z_offset = int(input_list[5]) # number of z-stacks above and below the center (z-slice showing the longest growth cone (typically R3 and/or R4))
		self.radius_expanse_ratio = int(input_list[6]) # max value of relative length given |C-T4| = 1


### get input
# inputs:
# annot_name = input_list[0]
# slice_type = input_list[1]
# num_angle_section = input_list[2]
# num_outside_angle = input_list[3]
# num_x_section = input_list[4]
# z_offset = input_list[5]
# radius_expanse_ratio = input_list[6]
# example of an input: annotation_example.csv, 1, 10, 5, 10, 10, 3
input_str = input()
input_list = input_str.split(', ')
print(input_str)

### get paths class
paths = Paths(input_list[0])

### get indexing class
matching_info = MatchingInfo()

### get analysis parameters
analysis_params_general = GeneralParams(input_list)