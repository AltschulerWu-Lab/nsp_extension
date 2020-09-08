# @Author: Weiyue Ji
# @Date:   2019-11-15 17:53:37
# @Last Modified by:   Weiyue Ji
# @Last Modified time: 2020-09-08 14:59:04
# Note: Jobs -1; ND acquisition - 2
#!/bin/sh
/Fiji.app/ImageJ-linux64 --system --headless -macro image_preprocessing_windowless.ijm 'folder1_lamina, folder2|1, 2|log_20200908.txt'