cd ~
git clone https://github.com/mariusgarenaux/fine_tuning_acronym
cd fine_tuning_acronym

git checkout formation-continue

BUCKET_PATH="../bucket/fine_tuning_acronym"

# fill bucket with empty folders for test, models and data
mkdir -p -v ${BUCKET_PATH}/data
mkdir -p -v ${BUCKET_PATH}/sessions

# copy base data
cp -i ./example_data/acronym.json ${BUCKET_PATH}/data/acronym.json


/opt/conda/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt