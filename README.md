# Quantitative Trading Strategies Final Project

[Description of the project here]

## Overview
**Strategy:**
Sign Trading (?)

## Data Sources



## Quick Start

To quickest way to run code in this repo is to use the following steps. First, you must have the `conda` package manager installed (e.g., via Anaconda).

Thenm open a terminal and navigate to the root directory of the project and create a 
conda environment using the following command:
```
conda create -n env_name_here python=3.12
conda activate env_name_here
```
and then install the dependencies with pip
```
pip install -r requirements.txt
```
Finally, you can then run 
```
doit
```
And that's it!


### General Directory Structure

 - Folders that start with `_` are automatically generated. The entire folder should be able to be deleted, because the code can be run again, which would again generate all of the contents. 

 - Anything in the `_data` folder (or your own RAW_DATA_DIR) or in the `_output` folder should be able to be recreated by running the code and can safely be deleted.

 - The `assets` folder is used for things like hand-drawn figures or other pictures that were not generated from code. These things cannot be easily recreated if they are deleted.

 - `_output` contains the .py generated from jupyter notebooks, and the jupyter notebooks with outputs, both in .md and in .html
 
 - `/src` contains the actual code. All notebooks in this folder will be stored cleaned from outputs (after running doit). That is in order to avoid unecessary commits from changes from simply opening or running the notebook.

 - The `data_manual` (DATA_MANUAL_DIR) is for data that cannot be easily recreated. 

 - `doit` Python module is used as the task runner. It works like `make` and the associated `Makefile`s. To rerun the code, install `doit` (https://pydoit.org/) and execute the command `doit` from the `src` directory. Note that doit is very flexible and can be used to run code commands from the command prompt, thus making it suitable for projects that use scripts written in multiple different programming languages.

 - `.env` file is the container for absolute paths that are private to each collaborator in the project. You can also use it for private credentials, if needed. It should not be tracked in Git.


#### Specific Files

- `pipeline.json`: is currently orchestrating the Chartbook, which will replace Sphinx. It will later be moved to pyproject.yoml.
- `dodo.py`: is the file that defines the tasks to be run by doit. It is the equivalent of a Makefile. It is the main entry point for running code in this project. It is also used to generate the documentation.
- `settings.py`: is the file that defines the settings for the project. It is the main entry point for running code in this project. It is also used to generate the documentation.
- `pyproject.toml`: is currently used to configure some 


### Data and Output Storage

I'll often use a separate folder for storing data. Any data in the data folder can be deleted and recreated by rerunning the PyDoit command (the pulls are in the dodo.py file). Any data that cannot be automatically recreated should be stored in the "data_manual" folder (or DATA_MANUAL_DIR).

Because of the risk of manually-created data getting changed or lost, I prefer to keep it under version control if I can.

Thus, data in the "_data" folder is excluded from Git (see the .gitignore file), while the "data_manual" folder is tracked by Git.

Output is stored in the "_output" directory. This includes dataframes, charts, and rendered notebooks. When the output is small enough, you can have it under version control to keep track of how dataframes change as analysis progresses, for example.

The _data directory and _output directory can be kept elsewhere on the machine. To make this easy, I always include the ability to customize these locations by defining the path to these directories in environment variables, which I intend to be defined in the `.env` file, though they can also simply be defined on the command line or elsewhere. The `settings.py` is reponsible for loading these environment variables and doing some like preprocessing on them.

The `settings.py` file is the entry point for all other scripts to these definitions. That is, all code that references these variables and others are loading by importing `config`.


### Naming Conventions

 - **`pull_` vs `load_`**: Files or functions that pull data from an external  data source are prepended with "pull_", as in "pull_fred.py". Functions that load data that has been cached in the "_data" folder are prepended with "load_".
 For example, inside of the `pull_CRSP_Compustat.py` file there is both a
 `pull_compustat` function and a `load_compustat` function. The first pulls from
 the web, whereas the other loads cached data from the "_data" directory.

