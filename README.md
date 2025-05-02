# Fine Tuning LLM


## Project structure

The project consists in several notebooks :

- [training.ipynb](training.ipynb) : contains the training script,

- [training_tools.py](training_tools.py) : contains helper functions used in the training,

- [create_dataset.ipynb](create_dataset.ipynb) : notebook to create a dataset from a list of acronyms and their definitions,

- [data_boost.py](data_boost.py) : helper connectors to create the dataset.

To run the project locally, install a .venv, and [requirements.txt](requirements.txt).

set datalab boolean to True in notebook; and checkpoint_path str

create a data folder with boosted_data.json

run cells


## Link to Hugging Face

To run the training, you need a HuggingFace account in order to retrieve base models from the hub. You also need to ask for an access to restriced models if you use one (p.e. Llama family).

You'll need to create an access token.

## Run with cloud computing

You can run the notebook locally, or distant in datalab-gcp or onyxia. To do so, import needed files in distant JupyterLab :

- [training.ipynb](training.ipynb),

- [training_tools.py](training_tools.py),

- [init_gcp.sh](init_gcp.sh) or [init_onyxia.sh](init_onyxia.sh),

- [data/boosted_data.json](data/boosted_data.json).

Then, open terminal in JupyterLab, and run the script [init_gcp.sh](./init_gcp.sh) (resp. [init_onyxia.sh](./init_onyxia.sh)) :

```bash
source init_gcp.sh
```

The script installs python libs used in the training notebook within the appropriate python (in gcp).

In the ends, it asks for a HuggingFace token to retrieve restricted models (like Llama - you need therefore to create a HuggingFace account and ask access for Llama models).
