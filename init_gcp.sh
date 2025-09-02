

cd ~
git clone https://github.com/mariusgarenaux/fine_tuning_acronym

# Creates a bucket for data, models and test results
# bucket path 

cd fine_tuning_acronym
BUCKET_PATH="../bucket/fine_tuning_acronym"

# fill bucket with empty folders for test, models and data
mkdir -p -v ${BUCKET_PATH}/data
mkdir -p -v ${BUCKET_PATH}/sessions

# copy base data
cp -i ./example_data/acronym.json ${BUCKET_PATH}/data/acronym.json


/opt/conda/bin/pip install -r 00-set_up/requirements.txt
/opt/conda/bin/python3 -c "from huggingface_hub import login; login()"