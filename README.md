# Fine Tuning LLM

This projects aims at fine-tuning a LLM in order to make it understand and memorize a list of given acronyms and their definitions. The true goal is to learn the different steps of fine-tuning a LLM for a given task.

The fine-tuning is split in 3 parts (each with its folder):

- First, we create a dataset using an instruct LLM in a distant infrastructure (with Open Web UI API),

- Then we use the hugging face [transformer](https://huggingface.co/docs/transformers/v4.17.0/en/index) library to load and train a LLM,

- Finally, we test the model using [sentence-transformer](https://www.sbert.net/) library or a LLM as a judge.

Alongside the notebooks, you will find some sandbox cells to manipulate objects (tokenizer, pre-trained model, dataset, tokens, ...). I advise you to try some of the examples to have a better understanding of the objects.

## Data

You'll see example data in the folder [example_data](example_data). These allows you to skip the first part [01-create_dataset](01-create_dataset).

The starting point is a file acronym.json, (see for example [acronym.json](example_data/acronym.json)); containing a list of acronym and their definitions. Using this file, we take advantage of a Large Language Model to create fake conversations about these acronyms. Those will be our training dataset. 

## Getting started

### Run on Datalab (GCP)

Upload the script [setup-fine_tuning.sh](setup-fine_tuning.sh) on JupyterLab, and run it from the terminal :

```bash
source setup-fine_tuning.sh
```

### Inference provider

In some parts (data-generation or llm-as-a-judge), you need to connect to a ollama server (local or distant).

To download it locally, run :

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Source

Training notebook and scripts were adapted from https://colab.research.google.com/drive/1DqKNPOzyMUXmJiJFvJITOahVDxCrA-wA#scrollTo=9Ixtdtpgyv_a; and hugging face documentation (p.e. https://huggingface.co/learn/llm-course/en/chapter11/3).