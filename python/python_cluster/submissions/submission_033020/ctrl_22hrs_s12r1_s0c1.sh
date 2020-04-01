# @Author: lily
# @Date:   2019-07-16 04:09:06
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-03-30 07:50:59
# @==========explanation========
# @inputs:
# @env = input_list[0]
# @fig_out_folder = input_list[1]
# @data_out_folder = input_list[2]
# @annotName = input_list[3]
# @slicetype = input_list[4], default = 0
# @centertype = input_list[5], default = 1
# @num_angleSection = input_list[6], default = 24
# @num_outsideAngle = input_list[7], default = 18
# @num_Xsection = input_list[8], default = 50
# @z_offset = input_list[9], default = 20
# @scale_factor = float(input_list[10]) = rfp/gfp
# @example input:
# @Euclid, FigureOutput_v033020, DataOutput_v033020, Ctrl_22hrs_Gal80_s3r1_annotation.csv, 0, 1, 24, 18, 50, 20, 0.69

#!/bin/sh
echo "Euclid, FigureOutput_v033020, DataOutput_v033020, Ctrl_22hrs_Gal80_s12r1_annotation.csv, 0, 1, 24, 18, 50, 20, 1.9" | /wynton/home/awlab/wji/anaconda3/bin/python3 /awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/main_functions/data_quantification_main_v0927.py

qstat -j $JOB_ID