[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

# StreamPort
Platform for multivariate data analysis, sensor networks and pipelines and device condition monitoring.
<div style="display: flex; justify-content: center;"> <img src="./sp_logo.png" alt="StreamPort Logo" width="200"></div>
<br>
<br>

# Setup
The streamPort is an experimental project under development. Below, we instruct how to setup a development environment.

### On Windows
- Install Python version 3.12 or above;
- Verify if the correct version is installed using `python --version` (it should be above 3.12);
- If you not detect python, add the path to the environmental veriables following https://realpython.com/add-python-to-path/;
- Check and note the python installation path with `(Get-Command python).Path`;
- Update pip with `python -m pip install --upgrade pip`;
- Check if pip version is 24 or above with `pip --version` and verify that the python path is a parent of the pip lib folder;
- If not already installed, install virtualenv with `pip install virtualenv`;
- You can make `pip list` to check is the virtualenv is installed;
- Clone the repository locally;
- From the local repository folder, start a virtual environment using `python -m venv env` (you can change `env` by other name of your preference);
- Then activate the virtual environment with `env/Scripts/activate.bat`, where `env` should be the name of the virtual environment you defined;
- Once the virtual environment is activated, you can install the required libraries with `pip install -r requirements.txt`;
- For the Jupiter Notebooks, you can select the kernel from the virtual environment;
- Run the `dev_core.ipynb` for testing the setup;

<br>
<br>

[Contact us](mailto:cunha@iuta.de) for questions or collaboration.
