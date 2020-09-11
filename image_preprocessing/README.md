# Image Preprocessing
This folder contains python scripts to batch process `.nd2` files obtained from a Nikon confocal microscopy. It will perform data format transformation (`.nd2` to `.tif`) and background subtraction.

## Description
- This code was developed to run on the UCSF Wynton cluster. However it can also run on any machine that have [ImageJ](https://imagej.nih.gov/ij/) installed. This code is intended to run windowlessly.
- The primary function of the code is to batch process multiple folders of images obtained from a Nikon confocal microscopy. It will perform data format transformation and background subtraction. 
    - `image_preprocessing_windowless.ijm`  is the main ImageJ macro file. It takes inputs (as string) about information of folders to be processed,  process them, and save the output images in the desired folder.
    - `prepross_submission.sh` is the bash script to run the macro file on a HPC cluster. However, you can also copy the content of the bash script and run the macro file locally using command line. 
