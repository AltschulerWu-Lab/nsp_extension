# image_quantification
This folder contains python scripts to transform annotated images using the standardized coordinate system described in the manuscript.

## Setup
This code was developed to run on the UCSF Wynton cluster. Running the code on a different system will require installation of the required packages and adapting the code accordingly. 

## Structure
- The "main" folder contains the main python script (`data_quantification_main.py`) to perform the entire image quantification process on one annotated image. 
- The "function" folder contains python scripts with functions to support the main script. If the sub-folder structure is kept intact, the functions should be imported automatically.
- The "submission" folder contains example of a bash script to run the analysis pipeline on UCSF Wynton cluster.
- The "data example" folder contains one example image and its annotations.
