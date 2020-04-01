# @Author: lily
# @Date:   2019-07-16 04:09:06
# @Last Modified by:   sf942274
# @Last Modified time: 2020-03-30 04:52:53
# @==========explanation========
# @inputs:
# @env = input_list[0]
# @fig_out_folder = input_list[1]
# @data_out_folder = input_list[2]
# @annot_name = input_list[3]
# @slice_type = input_list[4], default = 0
# @center_type = input_list[5], default = 1
# @num_angle_section = input_list[6], default = 24
# @num_outside_angle = input_list[7], default = 18
# @num_x_section = input_list[8], default = 40
# @z_offset = input_list[9], default = 20
# @scale_factor = int(input_list[10]) = rfp/gfp
# @example of an input: 
# @Windows, Data_Output, Figure_Output, Fz_26hrs_Gal80_s5r1_annotation.csv, 0, 1, 10, 5, 10, 10, 1

#!/bin/sh
echo "Euclid, FigureOutput_v033020, DataOutput_v033020, Ctrl_22hrs_Gal80_s2r1_annotation.csv, 0, 1, 24, 18, 50, 20, 0.62" | /wynton/home/awlab/wji/anaconda3/bin/python3 /awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/main_functions/data_quantification_main_v0927.py

qstat -j $JOB_ID