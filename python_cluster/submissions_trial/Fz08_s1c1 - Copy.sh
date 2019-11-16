# @Author: lily
# @Date:   2019-07-16 04:09:06
# @Last Modified by:   sf942274
# @Last Modified time: 2019-09-25 13:03:03
# @==========explanation========
# @inputs:
# @annotName = input_list[0]
# @slicetype = input_list[1]
# @centertype = input_list[2]
# @num_angleSection = input_list[3]
# @num_outsideAngle = input_list[4]
# @num_Xsection = input_list[5]
# @z_offset = input_list[6]

#!/bin/sh
echo "Fz_Gal80_s8r1_summary_old.csv, 1, 1, 24, 18, 40, 20, 1" | /netapp/home/wji/anaconda3/bin/python3 /awlab/projects/2019_09_NSP_Extension/code/python_cluster/main_functions/main_trial.py

qstat -j $JOB_ID