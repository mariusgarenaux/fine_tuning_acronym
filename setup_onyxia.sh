cd ${HOME}

INITIAL_PWD=$PWD

TP_DIR="$HOME/work"

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



# creates a kernel spec for this venv, and install it
mkdir -p /home/onyxia/ft_kernel
cat > /home/onyxia/ft_kernel/kernel.json <<EOL
{
  "argv": [
    "/home/onyxia/work/fine_tuning_acronym/.venv/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3 (ipykernel) FT",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "kernel_protocol_version": "5.5"
}
EOL

# finally install the kernel
jupyter kernelspec install /home/onyxia/ft_kernel --sys-prefix

# --------------------- install ollama ---------------------
sudo add-apt-repository universe --yes
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
cd ${INITIAL_PWD}