# @Author: Weiyue Ji
# @Date:   2019-11-15 17:53:37
# @Last Modified by:   sf942274
# @Last Modified time: 2020-02-07 17:54:58
# Note: Jobs -1; ND acquisition - 2
#!/bin/sh
/awlab/projects/2019_09_NSP_Extension/code/Fiji.app/ImageJ-linux64 --system --headless -macro /awlab/projects/2019_09_NSP_Extension/code/NSP_codes/image_preprocessing/imageJ_macros/processFoldersND2toTIF_windowless.ijm '20200126, 20200130, 20200201_Ni80_22hrs_slide0131_lamina, 20200201_Ni80_24hrs_slide0131_lamina|2, 2, 1, 1|log_v0207.txt'
