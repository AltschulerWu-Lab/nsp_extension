# Figure Generation
This folder contains python scripts and a jupyter notebook to generate figures in the manuscript.

## Structure
- _data_ folder contains all the data necessary to generate data figures in the manuscript.
- _settings.py_ and _summary_helper.py_ contains functions to support the notebook.
- All the plots are done in the notebook _generate_figures.ipynb_.


## Generate Figures
1. Paths

	paths are defined in the `Paths` class in _settings.py_. It contains paths to the processed data folder (`data_prefix`), desired output folder (`output_prefix`) and the code folder (`code_path`). It is currently set-up so that data and figure outputs are stored under _data_ and _figure_ sub-folder of the folder containing the notebook.

2. Output figure parameters

	In the "Golbal variables and parameters" block of the notebook, there are three parameters about output figures:
	  - `is_save`: if `True`, then when a graph is generated, it will be saved to the specified output folder.
	  - `fig_format`: format of the graph to be saved. Default is `svg`.
	  - `fig_res`: resolution of the graph to be saved. Default is 300.

3. Run code blocks to import data, generate graphs and compute statistical significance.
