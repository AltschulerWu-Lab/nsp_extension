# @Author: Weiyue Ji
# @Date:   2020-09-09 01:08:46
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-09 01:10:16

# @==========explanation========
# @inputs:
# @env = input_list[0] 
# @fig_out_folder = input_list[1] 
# @data_out_folder = input_list[2] 
# @annotName = input_list[3] 
# @slicetype = input_list[4]: 0-v1, 1-v2 
# @num_angleSection = input_list[5]: 30 
# @num_outsideAngle = input_list[6]: 30 
# @num_Xsection = input_list[7]: 50 
# @z_offset = input_list[8]: 20 
# @channels_type = input[9] - 'R3R4' or 'FasII' 

 #!/bin/sh 
echo "Euclid, Figure_output, Data_output, Ctrl_26hrs_Gal80_s1r1_example.csv, 0, 40, 40, 76, 20, R3R4" | /wynton/home/awlab/wji/anaconda3/bin/python3 /awlab/projects/2019_09_NSP_Extension/code/NSP_extension/python/python_cluster/main/data_quantification_main_v071501.py
qstat -j $JOB_ID