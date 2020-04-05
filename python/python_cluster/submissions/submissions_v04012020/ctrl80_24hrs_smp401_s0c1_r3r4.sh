# @Author: lily
# @Date:   2019-07-16 04:09:06
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-04-05 04:44:56
# @==========explanation========
# @inputs:
# @env = input_list[0]
# @fig_out_folder = input_list[1]
# @data_out_folder = input_list[2]
# @annotName = input_list[0]
# @slicetype = input_list[1]: 0-v1, 1-v2
# @centertype = input_list[2]: 0-T0, 1-center
# @num_angleSection = input_list[3]: 24
# @num_outsideAngle = input_list[4]: 18
# @num_Xsection = input_list[5]: 50
# @z_offset = input_list[6]: 20
# @scale_factor = input_list[7] = rfp/gfp
# @channels_type = input[11] - 'R3R4' or 'FasII'

#!/bin/sh
echo "Euclid, Figure_output_v0404, Data_output_v0404, Ctrl_24hrs_Gal80_s4r1_annotation.csv, 0, 1, 30, 30, 50, 20, 0.9, R3R4" | /wynton/home/awlab/wji/anaconda3/bin/python3 /awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/main/data_quantification_main_v040101.py

qstat -j $JOB_ID