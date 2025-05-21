# Fine Tuning LLM

This projects aims at fine-tuning a LLM in order to make it understand and memorize a list of given acronyms and their definitions. The true goal is to learn the different steps of fine-tuning a LLM for a given task.

The fine-tuning is split in 3 parts (each with its folder):

- First, we create a dataset using an instruct LLM in a distant infrastructure (with Open Web UI API),

- Then we use the hugging face [transformer](https://huggingface.co/docs/transformers/v4.17.0/en/index) library to load and train a LLM,

- Finally, we test the model using [sentence-transformer](https://www.sbert.net/) library or a LLM as a judge.

Alongside the notebooks, you will find some sandbox cells to manipulate objects (tokenizer, pre-trained model, dataset, tokens, ...). I advise you to try some of the examples to have a better understanding of the objects.

## Computing infrastructures

Whenever a model needs to be loaded (mostly training and test notebooks), you'll need computing infrastructure :

- Onyxia datalab : [https://datalab.sspcloud.fr](https://datalab.sspcloud.fr),

- Datalab of Rennes University (interacts with GCP),

- Locally if your machine has sufficient hardware (GPU / Apple Silicon Chip).

When loading the model is not necessary (for data-generation per example), you should be able to run notebooks locally, but remote is fine too. If they contain some inference parts, they are done remotely, using Open Web UI API. You'll need an access to an Open Web UI instance to use it (API token):

- if you have an account on [Onyxia](https://datalab.sspcloud.fr) (that includes ProConnect and hence Renater - any French University member can log in), you already have an Open Web UI access : https://llm.lab.sspcloud.fr,
    
- if you have an account on RAGaRenn [https://ragarenn.eskemm-numerique.fr)](https://ragarenn.eskemm-numerique.fr), you do have an access too,

- if neither of above is available for you, you can run an instance locally; or tweak the code of the notebooks to use an other inference provider (Ollama, OpenAI API, ...) - these features should be soon available.

## Project structure

The project consists in 3/4 parts :

- [00-set_up](00-set_up) : used to set up the training environment (either locally, datalab / google collab or onyxia) - you don't need to bother with what is inside this folder.

- [01-create_dataset](01-create_dataset) : here we boost the raw definition of acronyms into fake conversations using a LLM through OpenWeb UI (hosted either locally, in Onyxia or RagaRenn for example.)

- [02-fine_tune](02-fine_tune) : here we load the model and trains it on the above conversations

- [03-test](03-test) : here we load the previously trained model and test it ! We explore several strategy in order to compare models between them :

    - static embedding similarity between expected answers and fine-tuned model ones,

    - cross-encoder between expected answers and fine-tuned model ones,

    - using an other LLM to judge whether the fine-tuned model gives accurate definitions to acronyms in the dataset. 


## Data

You'll see example data in the folder [example_data](example_data). These allows you to skip the first part [01-create_dataset](01-create_dataset).

The starting point is a file acronym.json, (see for example [acronym.json](example_data/acronym.json)); containing a list of acronym and their definitions. Using this file, we take advantage of a Large Language Model to create fake conversations about these acronyms. Those will be our training dataset. 

## Getting started

Depending on where you run the notebooks, the set up is slightly different.

### Link to Hugging Face

To run the training - on whatever infrastructure, you'll need a HuggingFace account in order to retrieve models from the hub. You also need to ask for an access to restricted models if you use one (p.e. Llama family).

You'll need to create an access token, and declare it.

### Run on Onyxia

Nearly nothing to do, everything is pre-cooked for you !

Just go in Onyxia instance SSPCloud Datalab : [https://datalab.sspcloud.fr](https://datalab.sspcloud.fr).

Connect and create a vault named `fine_tuning` with following secrets :

![onyxia_vault_ex](00-set_up/onyxia_vault.png).

Then, simply click the following pre-cooked Onyxia Service :

> https://datalab.sspcloud.fr/launcher/ide/jupyter-pytorch-gpu?name=jupyter-pytorch-gpu&version=2.3.4&s3=region-ec97c721&init.personalInit=«https%3A%2F%2Fraw.githubusercontent.com%2Fmariusgarenaux%2Ffine_tuning_acronym%2Frefs%2Fheads%2Fmain%2F00-set_up%2Finit_onyxia.sh»&extraEnvVars[0].name=«WHICH_INFRA»&extraEnvVars[0].value=«onyxia»&extraEnvVars[1].name=«OWUI_FAV_MODEL»&extraEnvVars[1].value=«mistral-small3.1%3Alatest»&vault.secret=«fine_tuning»&autoLaunch=true

### Run Locally

Git clone the project 
Set up a python .venv, activate and install libraries :

> Supported with Python 3.12.9 - not tested with earlier versions (might cause problems with recent ML libs - transformers, ...)

```bash
git clone https://github.com/mariusgarenaux/fine_tuning_acronym
cd fine_tuning_acronym
python --version # check your python version
```

Go watch the [initialization script](00-set_up/init_locally.sh) and source it :

```bash
source 00-set_up/init_locally.sh
```

Finally, fill the config file with appropriate token and url at [conf/conf.yaml](conf/conf.yaml).

```yaml
OWUI_TOKEN: <your_owui_token>
OWUI_URL: https://<the_open_web_ui_instance_you_want_to_connect>/api/chat/completions
```

### Run on Datalab (GCP)

- in the terminal where you launched the datalab-client (you should have a SSH connection to the allocated VM), git clone the project :
```bash
git clone https://github.com/mariusgarenaux/fine_tuning_acronym
```

- then switch to JupyterLab, open a terminal

> - [optional but advised] run `bash` to have a non-archaic terminal

- go to project dir, look at [initialization script](00-set_up/init_gcp.sh) and source it :

```bash
cd fine_tuning_acronym
source 00-set_up/init_gcp.sh
```

- Fill the config file; as in [Run Locally](#run-locally).

## Start fine-tuning !

The notebooks are in order. Each one contains a README.md to help you understand what you are doing. to [create a dataset](01-create_dataset/create_dataset.ipynb). Then you can [fine-tune a model](02-fine_tune/training.ipynb). Finally, [test it](03-test).

## Source

Training notebook and scripts were adapted from https://colab.research.google.com/drive/1DqKNPOzyMUXmJiJFvJITOahVDxCrA-wA#scrollTo=9Ixtdtpgyv_a; and hugging face documentation (p.e. https://huggingface.co/learn/llm-course/en/chapter11/3).