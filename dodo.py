"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

import sys

sys.path.insert(1, "./src/")

import shutil
import shlex
from os import environ, getcwd, path
from pathlib import Path

import settings
from colorama import Fore, Style, init

# ====================================================================================
# PyDoit Formatting
# ====================================================================================

## Custom reporter: Print PyDoit Text in Green
# This is helpful because some tasks write to sterr and pollute the output in
# the console. I don't want to mute this output, because this can sometimes
# cause issues when, for example, LaTeX hangs on an error and requires
# presses on the keyboard before continuing. However, I want to be able
# to easily see the task lines printed by PyDoit. I want them to stand out
# from among all the other lines printed to the console.
from doit.reporter import ConsoleReporter

try:
    in_slurm = environ["SLURM_JOB_ID"] is not None
except:
    in_slurm = False


class GreenReporter(ConsoleReporter):
    def write(self, stuff, **kwargs):
        doit_mark = stuff.split(" ")[0].ljust(2)
        task = " ".join(stuff.split(" ")[1:]).strip() + "\n"
        output = (
            Fore.GREEN
            + doit_mark
            + f" {path.basename(getcwd())}: "
            + task
            + Style.RESET_ALL
        )
        self.outstream.write(output)


if not in_slurm:
    DOIT_CONFIG = {
        "reporter": GreenReporter,
        # other config here...
        # "cleanforget": True, # Doit will forget about tasks that have been cleaned.
        'backend': 'sqlite3',
        'dep_file': './.doit-db.sqlite'
    }
else:
    DOIT_CONFIG = {
        'backend': 'sqlite3',
        'dep_file': './.doit-db.sqlite'
    }
init(autoreset=True)

# ====================================================================================
# Configuration and Helpers for PyDoit
# ====================================================================================

BASE_DIR = Path(settings.BASE_DIR)
DATA_DIR = Path(settings.DATA_DIR)
RAW_DATA_DIR = Path(settings.RAW_DATA_DIR)
MANUAL_DATA_DIR = Path(settings.MANUAL_DATA_DIR)
OUTPUT_DIR = Path(settings.OUTPUT_DIR)
USER = settings.USER

## Helpers for handling Jupyter Notebook tasks
# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
def jupyter_execute_notebook(notebook):
    return f'jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --log-level WARN --inplace "{shlex.quote(f"./src/{notebook}.ipynb")}"'
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f'jupyter nbconvert --to html --log-level WARN --output-dir="{shlex.quote(str(output_dir))}" "{shlex.quote(f"./src/{notebook}.ipynb")}"'
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f'jupytext --to markdown --log-level WARN --output-dir="{shlex.quote(str(output_dir))}" "{shlex.quote(f"./src/{notebook}.ipynb")}"'
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f'jupyter nbconvert --log-level WARN --to python "{shlex.quote(f"./src/{notebook}.ipynb")}" --output "_{notebook}.py" --output-dir "{shlex.quote(str(build_dir))}"'
def jupyter_clear_output(notebook):
    return f'jupyter nbconvert --log-level WARN --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace "{shlex.quote(f"./src/{notebook}.ipynb")}"'
    # fmt: on


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################

def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [RAW_DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }

def task_pull_public_repo_data():
    """Pull public data from FRED and OFR API"""

    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_fred.py",
            "ipython ./src/pull_ofr_api_data.py",
        ],
        "targets": [
            RAW_DATA_DIR / "fred.parquet",
            RAW_DATA_DIR / "ofr_public_repo_data.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_fred.py",
            "./src/pull_ofr_api_data.py",
        ],
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    }

##############################$
## Demo: Other misc. data pulls
##############################$
# def task_pull_other():
#     """ """
#     file_dep = [
#         "./src/pull_bloomberg.py",
#         "./src/pull_CRSP_Compustat.py",
#         "./src/pull_CRSP_stock.py",
#         "./src/pull_fed_yield_curve.py",
#         ]
#     file_output = [
#         "bloomberg.parquet",
#         "CRSP_Compustat.parquet",
#         "CRSP_stock.parquet",
#         "fed_yield_curve.parquet",
#         ]
#     targets = [RAW_DATA_DIR / file for file in file_output]

#     return {
#         "actions": [
#             "ipython ./src/pull_bloomberg.py",
#             "ipython ./src/pull_CRSP_Compustat.py",
#             "ipython ./src/pull_CRSP_stock.py",
#             "ipython ./src/pull_fed_yield_curve.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": [],  # Don't clean these files by default.
#     }


# def task_summary_stats():
#     """ """
#     file_dep = ["./src/example_table.py"]
#     file_output = [
#         "example_table.tex",
#         "pandas_to_latex_simple_table1.tex",
#     ]
#     targets = [OUTPUT_DIR / file for file in file_output]

#     return {
#         "actions": [
#             "ipython ./src/example_table.py",
#             "ipython ./src/pandas_to_latex_demo.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


# def task_example_plot():
#     """Example plots"""

#     return {
#         "actions": [
#             # "date 1>&2",
#             # "time ipython ./src/example_plot.py",
#             "ipython ./src/example_plot.py",
#         ],
#         "targets": [
#             OUTPUT_DIR / "example_plot.png",
#         ],
#         "file_dep": [
#             "./src/example_plot.py",
#             "./src/pull_fred.py",
#         ],
#         "clean": True,
#     }


notebook_tasks = {
    # "01_example_notebook_interactive.ipynb": {
    #     "file_dep": [],
    #     "targets": [
    #         Path("./docs") / "01_example_notebook_interactive.html",
    #     ],
    # },
    # "02_example_with_dependencies.ipynb": {
    #     "file_dep": ["./src/pull_fred.py"],
    #     "targets": [
    #         Path(OUTPUT_DIR) / "GDP_graph.png",
    #         Path("./docs") / "02_example_with_dependencies.html",
    #     ],
    # },
}


def task_convert_notebooks_to_scripts():
    """Convert notebooks to script form to detect changes to source code rather
    than to the notebook's metadata.
    """
    build_dir = Path(OUTPUT_DIR)

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                jupyter_clear_output(notebook_name),
                jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [Path("./src") / notebook],
            "targets": [OUTPUT_DIR / f"_{notebook_name}.py"],
            "clean": True,
            "verbosity": 0,
        }


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                jupyter_execute_notebook(notebook_name),
                jupyter_to_html(notebook_name),
                copy_file(
                    Path("./src") / f"{notebook_name}.ipynb",
                    OUTPUT_DIR / f"{notebook_name}.ipynb",
                    mkdir=True,
                ),
                copy_file(
                    OUTPUT_DIR / f"{notebook_name}.html",
                    Path("./docs") / f"{notebook_name}.html",
                    mkdir=True,
                ),
                jupyter_clear_output(notebook_name),
                # jupyter_to_python(notebook_name, build_dir),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                OUTPUT_DIR / f"_{notebook_name}.py",
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                OUTPUT_DIR / f"{notebook_name}.ipynb",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

