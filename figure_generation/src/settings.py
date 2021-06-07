# -*- coding: utf-8 -*-
# @Author: Weiyue Ji
# @Date:   2020-03-27 15:06:33
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2021-06-06 17:44:19

import os

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
	def __init__(self):
		pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
		self.data_prefix = os.path.join(pardir, 'data')
		self.output_prefix = os.path.join(pardir, 'results')
		self.code_path = os.getcwd()

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

		# name of channels for each channels_type
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

### class that stores parameters for analysis
class GeneralParams:
	def __init__(self):
		self.slice_type = 0
		self.center_type = 0
		self.num_angle_section = 0
		self.num_outside_angle = 0
		self.num_x_section = 0
		self.z_offset = 0
		self.radius_expanse_ratio = 0

### class that stores parameters for analysis
paths = Paths()

### get indexing class
matching_info = MatchingInfo()

analysis_params_general = GeneralParams()