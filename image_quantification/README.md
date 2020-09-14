
# Image Quantification
This folder contains python scripts to transform annotated images using the standardized coordinate system described in the manuscript.

## Setup
This code was developed to run on the UCSF Wynton cluster. Running the code on a different system will require installation of the required packages and adapting the code accordingly. 

## Structure
- The `main` folder contains the main python script (`data_quantification_main.py`) to perform the entire image quantification process on one annotated image. 
- The `function` folder contains python scripts with functions to support the main script. If the sub-folder structure is kept intact, the functions should be imported automatically.
- The `submission` folder contains one example of bash script to run the quantification pipeline on UCSF Wynton cluster.
- The `data_example` folder contains one example image and its annotations.

## Image quantification process
**1. Image pre-processing and annotation**
- Images should convert to `tif` format in order to be imported by the python script. See folder `image_preprocessing` for custom ImageJ macro script to batch process `nd2` files output from NIS Elements software.
- Images were visually inspected, cropped and annotated using [Fiji](https://imagej.net/Fiji).  See the "Methods" section of the manuscript for detailed description of annotation. Annotation outputs include: 
	-  `zip` file: Fiji annotations. 
	- `csv` file: key parameters of the annotation: 1) X, Y, positions of heels. 2) major, minor axis lengths and the major axis angle of the target ellipse.

**2. Image quantification using `data_quantification_main.py`**
- Paths: paths are defined in the `Path` class in `settings.py` file in the `functions` folder. It contains paths to the input folder (`data_folder_path`), desired output folder (`output_folder_path`), folder to save log files (`log_folder_path`), and the code folder (`code_path`). Current folder structure is:
	- Inputs are stored under `data_example` folder located in the same sup-folder as `main` where the main python script is. Images are stored in the sub-folder `Images`, `csv` outputs from Fiji are stored in the sub-folder `ROIs`, and annotation of bundle and target indexing are stored in the sub-folder `Annotations`
	- Outputs are stored under `output_example` folder located in the same sup-folder as `main` where the main python script is. Data outputs will be stored under the sub-folder `Data_Output` while figures will be stored under the sub-folder `Figure_Output`.
	- Log files will be stored under `logs` folder located in the same sup-folder as `main` where the main python script is.
- Inputs:
	- String input: this will pass the name of the annotation file as well as key parameters to the script. Name and parameters are separated by comma. Example of a string input: `annotation_example.csv, 1, 10, 5, 10, 10, 3`. Parameters passed are: 
		- `slice_type`: define whether or not to force the line from center to T4 to 0 in defining relative angles. 
		- `num_angle_section`: define number of angle sections between T3 and T3'.
		- `num_outside_angle`: define number of angle sections to expand in calculation outside of angle(T3-c-T3').
		- `num_x_section` define number of length sections.
		- `z_offset`: define number of z-stacks above and below the center (z-slice showing the longest growth cone).
		- `radius_expanse_ratio`: define max value of relative length given |C-T4| = 1.
	- Image in `tif` format
	- Output from Fiji annotation in `csv` format.
	- Annotation about bundle and its putative target mapping in `csv` format.
- Outputs:
	- Data output: `pickle` format. This is a dictionary that contains the calculated density maps of each bundles, as well as key parameters.
	- Figure outputs: Figures of the density maps.
