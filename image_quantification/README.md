
# image_quantification
This folder contains python scripts to transform annotated images using the standardized coordinate system described in the manuscript.

## Setup
- This code was developed to run on the UCSF Wynton cluster. Running the code on a different system will require installation of the required packages and adapting the code accordingly. 
- Dependencies:
	- Python >= 3.6
	- seaborn >= 0.11.0
	- scipy >= 1.5.0
	- matplotlib >= 3.2.2
	- pandas >= 1.0.5
	- scikit_image >= 0.16.2
	- numpy >= 1.18.5

## Structure
- The _src_ folder contains all the python scripts to perform the image quantification process.
	+ The _main_ folder contains the main python script (_data_quantification_main.py_) to perform the entire image quantification process on one annotated image. 
	+ The _function_ folder contains python scripts with functions to support the main script. If the sub-folder structure is kept intact, the functions should be imported automatically.
	+ The _submission_ folder contains one example of bash script to run the quantification pipeline on UCSF Wynton cluster.
- The _data_ folder contains one example image and its annotations.

## Image quantification process
**1. Image pre-processing and annotation**
- Images should convert to TIFF format in order to be imported by the python script. See folder _image_preprocessing_ for custom ImageJ macro script to batch process ND2 files output from Nikon Elements software.
- Images were visually inspected, cropped and annotated using [Fiji](https://imagej.net/Fiji).  See the "Methods" section of the manuscript for detailed description of annotation. Annotation outputs include: 
	-  zip file: Fiji annotations. 
	- csv file: key parameters of the annotation: 1) X, Y, positions of heels. 2) major, minor axis lengths and the major axis angle of the target ellipse.

**2. Image quantification using _data_quantification_main.py_**
- Paths: paths are defined in the `Path` class in _settings.py_ file in the _functions_ folder. It contains paths to the input folder (`data_folder_path`), desired output folder (`output_folder_path`), folder to save log files (`log_folder_path`), and the code folder (`code_path`). Current folder structure is:
	- Inputs are stored under _data_ folder located in the same sup-folder as _src_ where all python scripts are. Images are stored in the sub-folder _Images_, csv outputs from Fiji are stored in the sub-folder _ROIs_, and annotation of bundle and target indexing are stored in the sub-folder _Annotations_
	- Outputs will be stored under _output_ folder located in the same sup-folder as _src_ where all python scripts are. Data outputs will be stored under the sub-folder _Data_Output_ while figures will be stored under the sub-folder _Figure_Output_.
	- Log files will be stored under _/doc/logs_ folder.
- Inputs:
	- String input: this will pass the name of the annotation file as well as key parameters to the script. Name and parameters are separated by comma. Example of a string input: `annotation_example.csv, 1, 40, 40, 76, 20, 3.8`. Parameters passed are: 
		- `slice_type`: define whether or not to force the line from center to T4 to 0 in defining relative angles. 
		- `num_angle_section`: define number of angle sections between T3 and T3'. Must be an even number.
		- `num_outside_angle`: define number of angle sections to expand in calculation outside of angle(T3-c-T3').
		- `num_x_section` define number of length sections.
		- `z_offset`: define number of z-stacks above and below the center (z-slice showing the longest growth cone).
		- `radius_expanse_ratio`: define max value of relative length given |C-T4| = 1.
	- Image in TIFF format
	- Output from Fiji annotation in CSV format.
	- Annotation about bundle and its putative target mapping in CSV format.
- Outputs:
	- Data output: pickle format. This is a dictionary that contains the calculated density maps of each bundles, as well as key parameters.
	- Figure outputs: Figures of the density maps.
