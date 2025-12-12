[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

# StreamPort

<p align="center" width="100%">
<img width="40%" src="sp_logo.png" alt="Logo" />
</p>

Platform for multivariate data Analyses, sensor networks and pipelines and device condition monitoring and diagnostics.

# Setup
The streamPort is an experimental project under development. Below, we instruct you on how to setup a development environment. Note that certain packages included with this release require Chrome or any chromium-based browser to enable smooth function. Install Chrome via https://www.google.com/chrome/.

## On Windows
- Install Python version 3.12 or above;
- Verify if the correct version is installed using `python --version` (it should be above 3.12);
- If you cannot detect python, add the path to the environment variables following https://realpython.com/add-python-to-path/;
- Check and note the python installation path with `(Get-Command python).Path`;
- Update pip with `python -m pip install --upgrade pip`;
- Check if pip version is 24 or above with `pip --version` and verify that the python path is a parent of the pip lib folder;
- If not already installed, install virtualenv with `pip install virtualenv`;
- You can run `pip list` to check if the virtualenv is installed;
- Clone the repository locally;
- From the local repository folder, start a virtual environment using `python -m venv env` (you can change `env` to another name of your preference);
- Then activate the virtual environment with `env/Scripts/activate.bat`, where `env` should be the name of the virtual environment you defined;
- Once the virtual environment is activated, you can install the required libraries with `pip install -r requirements.txt`;
- For example workflows using Jupyter Notebooks, see the `notebooks` directory. 
- For the Jupyter Notebooks, you can select the kernel from the virtual environment;
- Run the `dev_core.ipynb` for testing the setup;

<br>

# Streamlit App
- Install Streamlit version 1.48.1 or above (included in requirements);
- Configure the page layout/design by editing the `.streamlit/config.toml` file;
- From the root directory, make all modules in the current path discoverable by setting PYTHONPATH:;

## CMD
- `set PYTHONPATH=.`;
- Run app using `streamlit run src/StreamPort/app/Home.py`;

<br>

# Build Library
- The build library is required to build the libraries. It can be installed with the command `pip install build`;
- The library can be built by first navigating to the root directory `StreamPort` and then running `py -m build`, which creates the `dist` directory containing the `.whl` and `.tar.gz` files for installation;

# Package Installation
- The StreamPort package can be installed by first navigating to the `dist` directory;
- Then, to install the package, run `pip install streamport-0.0.1-py3-none-any.whl`, or the `.whl` file that is created on build;
- To install and make source code editable, run from root directory `pip install -e .`;
- Now the StreamPort package can be imported using `pip import StreamPort`, and all its public classes can now be used using `from StreamPort import *`, where * includes `device` and `machine_learning`;
- Your first `Device Analysis` class object can be created using `pc = device.analyses.PressureCurvesAnalyses(<files>)` or `ms = device.analyses.MassSpecAnalyses(<files>)`, where files is the file or list of files to be analyzed, and similarly for `Machine Learning Analysis` or `Methods` and so on (see notebooks directory for example usages);

<br>

[Contact us](mailto:cunha@iuta.de) for questions or collaboration.
