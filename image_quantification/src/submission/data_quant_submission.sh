# @Author: Weiyue Ji
# @Date:   2020-09-09 01:08:46
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2021-06-06 18:00:04

# @==========explanation========
# @inputs:
# @annotName = input_list[0] 
# @slicetype = input_list[1]: 0-v1, 1-v2 
# @num_angleSection = input_list[2]: 40, must be an even number.
# @num_outsideAngle = input_list[3]: 40 
# @num_Xsection = input_list[4]: 76 
# @z_offset = input_list[5]: 20
# @radius_expanse_ratio = input_list[6]: 3.8

 #!/bin/sh 
echo "annotation_example.csv, 1, 40, 40, 76, 20, 3.8" | python3 data_quantification_main.py
qstat -j $JOB_ID