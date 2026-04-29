# Fine Tuning LLM

This projects aims at fine-tuning a LLM in order to make it understand and memorize a list of given acronyms and their definitions. The true goal is to learn the different steps of fine-tuning a LLM for a given task.

The fine-tuning is split in 3 parts (each with its folder):

- First, we create a dataset using an instruct LLM in a distant infrastructure (with Open Web UI API),

- Then we use the hugging face [transformer](https://huggingface.co/docs/transformers/v4.17.0/en/index) library to load and train a LLM,

- Finally, we test the model using [sentence-transformer](https://www.sbert.net/) library or a LLM as a judge.

Alongside the notebooks, you will find some sandbox cells to manipulate objects (tokenizer, pre-trained model, dataset, tokens, ...). I advise you to try some of the examples to have a better understanding of the objects.

All notebooks don't work 'as is', but must be completed by replacing '...' with real code. The cells that has to be completed starts by `#TO COMPLETE`.

> You will find correction of each notebook, in the form 'completed\_<notebook_name>.ipynb'.

## Data

You'll see example data in the folder [example_data](example_data). These allows you to skip the first part [01-create_dataset](01-create_dataset).

The starting point is a file acronym.json, (see for example [acronym.json](example_data/acronym.json)); containing a list of acronym and their definitions. Using this file, we take advantage of a Large Language Model to create fake conversations about these acronyms. Those will be our training dataset.

## Getting started

Several datalab-like infrastructures are supported. The uv.lock file (`uv sync`) is supported only for Onyxia, because it downloads torch for CUDA 12.6

### Run on Onyxia (authentication needed through ProConnet or RENATER)

Just click on the link below. Launching the container and installing all packages can be quite long (up to 5 minutes) :

https://datalab.sspcloud.fr/launcher/ide/jupyter-python-gpu?name=tp-fine-tuning&version=2.4.6&s3=region-79669f20&persistence.size=«30Gi»&init.personalInit=«https%3A%2F%2Fraw.githubusercontent.com%2Fmariusgarenaux%2Ffine_tuning_acronym%2Frefs%2Fheads%2Fformation-continue%2Fsetup_onyxia.sh»&autoLaunch=true

You should have access to a Jupyter Lab with the TP installed on it.

> **IMPORTANT** : Run all notebooks by choosing the kernel from '/home/onyxia/work/fine_tuning_acronym/.venv', otherwise you will get ModuleNotFount errors !

### Run on Datalab (GCP), or locally

Upload the script [setup_datalab.sh](setup-fine_tuning.sh) on JupyterLab, and run it from the terminal (! it will create a directory named 'bucket' on the parent of the repo !):

```bash
source setup_datalab.sh
```

Or simply run the bash one liner:

```bash
curl -fsSL https://raw.githubusercontent.com/mariusgarenaux/fine_tuning_acronym/refs/heads/formation-continue/setup-fine_tuning.sh | sh
```

Then you need to install ollama :

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

And then start the ollama server :

```bash
ollama serve
```

Finally pull model for inference, for example :

```bash
ollama pull gemma3:4b
```

## Source

Training notebook and scripts were adapted from https://colab.research.google.com/drive/1DqKNPOzyMUXmJiJFvJITOahVDxCrA-wA#scrollTo=9Ixtdtpgyv_a; and hugging face documentation (p.e. https://huggingface.co/learn/llm-course/en/chapter11/3).
