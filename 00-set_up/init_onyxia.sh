#!/bin/bash

# This init script clones a Git repository that contains a Jupyter notebook
# named `tutorial.ipynb` and opens it in Jupyter Lab at startup
# Expected parameters : None

# Clone repository and give permissions to the onyxia user
GIT_REPO=fine_tuning_acronym
git clone  https://github.com/mariusgarenaux/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/

# Install additional packages if a requirements.txt file is present in the project
REQUIREMENTS_FILE=${GIT_REPO}/00-set_up/requirements.txt
[ -f $REQUIREMENTS_FILE ] && pip install -r $REQUIREMENTS_FILE

# Add dark theme by default
mkdir -p /home/onyxia/.jupyter/lab/user-settings/@jupyterlab/apputils-extension 
cp -i ${GIT_REPO}/00-set_up/themes.jupyterlab-settings /home/onyxia/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# Creates a bucket for data, models and test results
mkdir -p ${GIT_REPO}/bucket/data/batched_data
mkdir -p ${GIT_REPO}/bucket/models
mkdir -p ${GIT_REPO}/bucket/tests

# Copy acronym.json 
cp -i ${GIT_REPO}/example_data/acronym.json ${GIT_REPO}/bucket/data/acronym.json

# Open the relevant notebook when starting Jupyter Lab
echo "c.LabApp.default_url = '/lab/tree/${GIT_REPO}/README.md'" >> /home/onyxia/.jupyter/jupyter_server_config.py

