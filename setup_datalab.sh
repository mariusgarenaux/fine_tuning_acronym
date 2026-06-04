# ----- for datalab (https://datalab.univ-rennes.fr/) ----

cd ${HOME}
BUCKET_PATH="$HOME/bucket/fine_tuning_acronym"


# fill bucket with empty folders for test, models and data
mkdir -p -v ${BUCKET_PATH}/data
mkdir -p -v ${BUCKET_PATH}/sessions

# copy base data
cp -i ${HOME}/fine_tuning_acronym/example_data/acronym.json ${BUCKET_PATH}/data/acronym.json

# Check if ollama is installed
if ! command -v ollama &> /dev/null
then
    echo "ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "ollama is already installed."
fi
ollama start &

# Give the server some time to start 
sleep 5

# pull model
ollama pull qwen3:1.7b

# Stop existing jupyter lab on port 8888
# fuser -k 8888/tcp || true

# runs the jupyter lab server
/root/fine_tuning_acronym/.venv/bin/jupyter-lab --allow-root --IdentityProvider.token=token --ServerApp.allow_remote_access=True --ServerApp.base_url=/notebook --NotebookApp.token=token
