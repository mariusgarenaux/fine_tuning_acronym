cd ~
git clone https://github.com/mariusgarenaux/fine_tuning_acronym
cd fine_tuning_acronym

git checkout formation-continue

BUCKET_PATH="~/bucket/fine_tuning_acronym"

# fill bucket with empty folders for test, models and data
mkdir -p -v ${BUCKET_PATH}/data
mkdir -p -v ${BUCKET_PATH}/sessions

# copy base data
cp -i ~/fine_tuning_acronym/example_data/acronym.json ${BUCKET_PATH}/data/acronym.json

python -m venv .venv
source .venv/bin/activate

pip install .


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

# Give the server some time to start (adjust as necessary)
sleep 5

# Pull the Gemma 3 model
echo "Pulling Gemma 3 model..."
ollama pull gemma3:4b

echo "Ollama server started and Gemma 3 model pulled."