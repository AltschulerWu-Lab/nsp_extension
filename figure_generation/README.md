# figure_generation
This folder contains python scripts and a jupyter notebook to generate figures in the manuscript.

## Structure
- `data` folder contains all the data necessary to generate data figures in the manuscript.
- `settings.py` and `summary_helper.py` contains functions to support the notebook.
- All the plots are done in the notebook `generate_figures.ipynb`.


## `generate_figures.ipynb`
- paths are defined in the `Paths` class in `settings.py`. It contains paths to the processed data folder (`data_prefix`), desired output folder (`output_prefix`) and the code folder (`code_path`). It is currently set-up so that the current sub-folder structures are kept intact.
- In the "Golbal variables and parameters" block of the notebook, there are three parameters about output figures:
  - `is_save`: if `True`, then when a graph is generated, it will be saved to the specified output folder.
  - `fig_format`: format of the graph to be saved. Default is `svg`.
  - `fig_res`: resolution of the graph to be saved. Default is 300.
- Run code blocks to import data, generate graphs and compute statistical significance.
