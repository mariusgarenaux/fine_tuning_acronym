#!/bin/bash

# This init script clones a Git repository that contains a Jupyter notebook
# named `tutorial.ipynb` and opens it in Jupyter Lab at startup
# Expected parameters : None

# Clone repository and give permissions to the onyxia user
GIT_REPO=fine_tuning_acronym
git clone  https://github.com/mariusgarenaux/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/

# Install additional packages if a requirements.txt file is present in the project
REQUIREMENTS_FILE=${GIT_REPO}/requirements.txt
[ -f $REQUIREMENTS_FILE ] && pip install -r $REQUIREMENTS_FILE

# Add dark theme by default
mkdir -p /home/onyxia/.jupyter/lab/user-settings/@jupyterlab/apputils-extension 
cp ${GIT_REPO}/themes.jupyterlab-settings /home/onyxia/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# Open the relevant notebook when starting Jupyter Lab
echo "c.LabApp.default_url = '/lab/tree/${GIT_REPO}/README.md'" >> /home/onyxia/.jupyter/jupyter_server_config.py

# python3 -c "from huggingface_hub import login; login()"
# paste token and say n
