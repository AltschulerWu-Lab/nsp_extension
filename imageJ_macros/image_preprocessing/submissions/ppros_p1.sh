# @Author: Weiyue Ji
# @Date:   2019-11-15 17:53:37
# @Last Modified by:   sf942274
# @Last Modified time: 2020-03-21 03:15:26
# Note: Jobs -1; ND acquisition - 2
#!/bin/sh
/awlab/projects/2019_09_NSP_Extension/code/Fiji.app/ImageJ-linux64 --system --headless -macro /awlab/projects/2019_09_NSP_Extension/code/NSP_extension/imageJ_macros/image_preprocessing/imageJ_macros/processFoldersND2toTIF_windowless.ijm '20200318, 20200318_Ni80_30hrs_slide0317_lamina, 20200318_Ni80_32hrs_slide0317_lamina, 20190412_362x60_3color_25C_2022hrs_lamina, 20190414_32x70_25C_24hrs_lamina|2, 1, 1, 2, 2|log_v0318_p1.txt'