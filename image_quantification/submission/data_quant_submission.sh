# @Author: Weiyue Ji
# @Date:   2020-09-09 01:08:46
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-10 16:24:28

# @==========explanation========
# @inputs:
# @annotName = input_list[1] 
# @slicetype = input_list[2]: 0-v1, 1-v2 
# @num_angleSection = input_list[3]: 30 
# @num_outsideAngle = input_list[4]: 30 
# @num_Xsection = input_list[5]: 50 
# @z_offset = input_list[6]: 20 

 #!/bin/sh 
echo "annotation_example.csv, 0, 40, 40, 76, 20" | python3 data_quantification_main.py
qstat -j $JOB_ID