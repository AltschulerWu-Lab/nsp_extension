# @Author: lily
# @Date:   2019-07-16 04:09:06
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-03-30 05:56:21
# @==========explanation========
# @inputs:
# @annotName = input_list[0]
# @slicetype = input_list[1]
# @centertype = input_list[2]
# @num_angleSection = input_list[3]
# @num_outsideAngle = input_list[4]
# @num_Xsection = input_list[5]
# @z_offset = input_list[6]
# @scale_factor = int(input_list[7])

#!/bin/sh
echo "Euclid, FigureOutput_v033020, DataOutput_v033020, Ctrl_22hrs_Gal80_s3r1_annotation.csv, 0, 1, 24, 18, 50, 20, 0.69" | /wynton/home/awlab/wji/anaconda3/bin/python3 /awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/main_functions/data_quantification_main_v0927.py

qstat -j $JOB_ID