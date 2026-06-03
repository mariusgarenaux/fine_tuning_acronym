# ----- for datalab (https://datalab.univ-rennes.fr/) ----

cd ${HOME}

# --------------------- cloning repo ---------------------

git clone https://github.com/mariusgarenaux/fine_tuning_acronym ${HOME}/fine_tuning_acronym
cd ${HOME}/fine_tuning_acronym

git checkout formation-continue

BUCKET_PATH="$HOME/bucket/fine_tuning_acronym"

# fill bucket with empty folders for test, models and data
mkdir -p -v ${BUCKET_PATH}/data
mkdir -p -v ${BUCKET_PATH}/sessions

# copy base data
cp -i ${HOME}/fine_tuning_acronym/example_data/acronym.json ${BUCKET_PATH}/data/acronym.json


# # # ----------- install the venv and sync with uv -----------
# uv venv ${HOME}/fine_tuning_acronym/.venv
# source ${HOME}/fine_tuning_acronym/.venv
# uv sync
# deactivate



# -- creates a kernel spec for this venv, and install it --
# mkdir -p ${HOME}/ft_kernel
# cat > ${HOME}/ft_kernel/kernel.json <<EOL
# {
#   "argv": [
#     "$HOME/fine_tuning_acronym/.venv/bin/python",
#     "-m",
#     "ipykernel_launcher",
#     "-f",
#     "{connection_file}"
#   ],
#   "display_name": "Python 3 ($HOME/fine_tuning_acronym/.venv)",
#   "language": "python",
#   "metadata": {
#     "debugger": true
#   },
#   "kernel_protocol_version": "5.5"
# }
# EOL

# finally install the kernel
# jupyter kernelspec install ${HOME}/ft_kernel --sys-prefix

# --------------------- install ollama ---------------------
#sudo add-apt-repository universe --yes
# apt-get install zstd

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

# Pull the Qwen3.5:4b model
echo "Pulling Qwen3:4b model..."
ollama pull qwen3:4b

echo "Ollama server started and Qwen3:4b model pulled."


# ------------------- set up chatbot ----------------------

# go back to initial PWD
cd ${HOME}