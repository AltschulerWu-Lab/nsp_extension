# @Author: Weiyue Ji
# @Date:   2019-11-15 17:53:37
# @Last Modified by:   sf942274
# @Last Modified time: 2020-03-18 18:22:55
# Note: Jobs -1; ND acquisition - 2
#!/bin/sh
/awlab/projects/2019_09_NSP_Extension/code/Fiji.app/ImageJ-linux64 --system --headless -macro /awlab/projects/2019_09_NSP_Extension/code/NSP_extension/imageJ_macros/image_preprocessing/imageJ_macros/processFoldersND2toTIF_windowless.ijm '20191208_Fz_full_42hrs_slide1208_lamina, 20191211_Fz_full_28hrs_slide1208_lamina, 20191211_Fz_full_34hrs_slide1208_lamina, 20191214_Fz80_38plus_slide1214_lamina, 20191216_Fz_full_4046hrs_slide1216_lamina, 20191230_Ctrl_full_slide1229_eye, 20200116_CtrlF_4620hrs_slide0117_eye, 20200116_Fz_2024hrs_slide0116_eye|1, 1, 1, 1, 1, 1, 1, 1|log_v318_p2.txt'
