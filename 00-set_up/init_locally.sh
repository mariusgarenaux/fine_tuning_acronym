
# Creates a bucket for data, models and test results
# bucket path 

BUCKET_PATH="./bucket"

# fill bucket with empty folders for test, models and data
mkdir -p -v ${BUCKET_PATH}/data/batched_data
mkdir -p -v ${BUCKET_PATH}/models
mkdir -p -v ${BUCKET_PATH}/tests

# pre-cook conf file
cp -i ./conf/example_conf.yaml ./conf/conf.yaml

# copy acronym data
cp -i ./example_data/acronym.json ${BUCKET_PATH}/data/acronym.json


python -m venv .venv
source .venv/bin/activate
pip install -r 00-set_up/requirements.txt
python3 -c "from huggingface_hub import login; login()"