# ----- for datalab (https://datalab.univ-rennes.fr/) ----

cd ${HOME}

INITIAL_PWD=$PWD

TP_DIR="$HOME/myhomedir"

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


# ----------------------- install uv ----------------------
curl -LsSf https://astral.sh/uv/install.sh | sh


# ----------- install the venv and sync with uv -----------
/root/.local/bin/uv venv
source .venv/bin/activate
/root/.local/bin/uv sync
deactivate
cd ${HOME}



# -- creates a kernel spec for this venv, and install it --
mkdir -p ${TP_DIR}/ft_kernel
cat > ${TP_DIR}/ft_kernel/kernel.json <<EOL
{
  "argv": [
    "$TP_DIR/fine_tuning_acronym/.venv/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3 ($TP_DIR/fine_tuning_acronym/.venv)",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "kernel_protocol_version": "5.5"
}
EOL

# finally install the kernel
cp -r ${TP_DIR}/ft_kernel /usr/local/share/jupyter/kernels/ft_kernel

# --------------------- install ollama ---------------------
#sudo add-apt-repository universe --yes
apt-get install zstd

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

# install libs
pip3.12 install jupyterlab-miami-nights
pip3.12 install pydantic_ai_kernel
pip3.12 install ipywidgets
# pip3.12 install jupyter-mcp
# pip3.12 install "jupyter-collaboration==4.0.2"
# pip3.12 install "jupyter-mcp-tools>=0.1.4"

# update to python3.12 in kernelspec
cat > /usr/local/share/jupyter/kernels/pydantic_ai/kernel.json <<EOL
{"argv": ["python3.12", "-m", "pydantic_ai_kernel", "-f", "{connection_file}"], "display_name": "Pydantic AI Agent", "interrupt_mode": "message", "language": "text", "env": {"JUPYTER_SERVER_URL": "${JUPYTER_SERVER_URL%/}"}}
EOL

# set up config
cat <<'EOF' > /root/.jupyter/jupyter_pydantic_ai_config.yaml
agent_name: coder
system_prompt: "You are an AI assistant designed to provide concise, accurate,
              and relevant information. Respond directly to user queries while ensuring
              clarity and understanding. Engage users in a conversational manner,
              demonstrating empathy and adaptability to their needs. Avoid unnecessary
              details, repetition, or embellishments, and focus on delivering
              solutions efficiently."
model:
  model_name: qwen3:4b
  model_type: openai
  model_provider:
    name: ollama
    params:
      base_url: http://127.0.0.1:11434/v1
mcp_servers:
  jupyter:
    command: /root/.local/bin/uvx
    args:
      - jupyter-mcp-server@latest
    env:
      JUPYTER_URL: "${JUPYTER_SERVER_URL}"
      JUPYTER_TOKEN: "token"
      ALLOW_IMG_OUTPUT: "true"
mcp_servers_user_approval:
  jupyter:
    jupyter_list_files: false
    jupyter_list_kernels: false
    jupyter_use_notebook: true
    jupyter_list_notebooks: false
    jupyter_restart_notebook: false
    jupyter_unuse_notebook: false
    jupyter_read_notebook: false
    jupyter_insert_cell: true
    jupyter_overwrite_cell_source: true
    jupyter_edit_cell_source: true
    jupyter_insert_execute_code_cell: true
    jupyter_execute_cell: true
    jupyter_read_cell: false
    jupyter_delete_cell: true
    jupyter_move_cell: true
    jupyter_execute_code: true
    jupyter_connect_to_jupyter: true
display_thinking: True
formatter: md
use_widget: True
EOF

# go back to initial PWD
cd ${INITIAL_PWD}   