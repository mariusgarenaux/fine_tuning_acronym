cd ${HOME}


if [ "$1" == "ONYXIA" ]; then
    echo "Installing TP on ~/work"
    TP_DIR="$HOME/work"
    PYTHON_DIR="/opt/python/bin/python"
else
    echo "Installing TP on ~"
    TP_DIR="$HOME"
fi


# --------------------- cloning repo ---------------------

git clone https://github.com/mariusgarenaux/fine_tuning_acronym ${TP_DIR}/fine_tuning_acronym
cd ${TP_DIR}/fine_tuning_acronym

git checkout formation-continue

BUCKET_PATH="$TP_DIR/bucket/fine_tuning_acronym"

# fill bucket with empty folders for test, models and data
mkdir -p -v ${BUCKET_PATH}/data
mkdir -p -v ${BUCKET_PATH}/sessions

# copy base data
cp -i ${TP_DIR}/fine_tuning_acronym/example_data/acronym.json ${BUCKET_PATH}/data/acronym.json


# ----------- install the venv and sync with uv -----------
uv venv
source .venv/bin/activate
uv sync
deactivate
cd ${HOME}



# TODO : add the kernel creator and installer,
# that allows to select a kernel from notebooks inside jupyter
# lab

# --------------------- install ollama ---------------------
sudo add-apt-repository universe
sudo apt-get install zstd

# Check if ollama is installed
if ! command -v ollama &> /dev/null
then
    echo "ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "ollama is already installed."
fi

# Start ollama server in the background
echo "Starting ollama server..."
ollama start &

# Give the server some time to start 
sleep 5

# Pull the Gemma 3 model
echo "Pulling Gemma 3 model..."
ollama pull gemma3:4b

echo "Ollama server started and Gemma 3 model pulled."


# go back to initial PWD
cd ${PWD}