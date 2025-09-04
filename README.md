# Fine Tuning LLM

This projects aims at fine-tuning a LLM in order to make it understand and memorize a list of given acronyms and their definitions.

We use the frameworks [mlflow](https://mlflow.org/) and [metaflow](https://metaflow.org/). _Metaflow_ as a workflow management system, and _mlflow_ to keep track of previous runs.

The workflow uses the hugging face [transformer](https://huggingface.co/docs/transformers/v4.17.0/en/index) library to load and train a LLM.
The next step consist in testing a model using [sentence-transformer](https://www.sbert.net/) library for output similarities, as well a LLM as a judge (called _via_ an OpenWebUI API - here [RAGaRenn](https://ragarenn.eskemm-numerique.fr/index.html)).

You can access a smaller version of this project by going in the branch _formation-continue_. 

## Project structure 

- [main.py](main.py) contains the workflow, (see [metaflow](https://metaflow.org/) documentation). It can be started with : `python mlflow.py run`.

- [programmer.py](programmer.py) file to make multiple runs of the workflow, with different _configurations_. It can be adapted easily.

- [config_loader.py](config_loader.py) loads the config file with pydantic checks, for the workflow.

- [tools.py](tools.py) contains diverses tools used in the workflow, like an OpenWebUIConnector to make authenticated calls to the OpenWebUI API.

## Data

You'll see example data in the folder [example_data](example_data). The training data (as well as test one) contains conversations about acronyms and their definitions. You can use any dataset with the same structure. If you want to generate data from a list acronym, you can take a look on the simplified version of this project, on the branch _formation-continue_ : [https://github.com/mariusgarenaux/fine_tuning_acronym/tree/formation-continue](https://github.com/mariusgarenaux/fine_tuning_acronym/tree/formation-continue).

## Link to Hugging Face

If you finetune a model that need a HuggingFace authentication, you have to give your HuggingFace token by running in a terminal : 

```
python -c "from huggingface_hub import login; login()"
```

## Getting started

Git clone the project,
Set up a python .venv, activate and install libraries :

> Supported with Python 3.12.9 - not tested with earlier versions (might cause problems with recent ML libs - transformers, ...)

```bash
git clone https://github.com/mariusgarenaux/fine_tuning_acronym
cd fine_tuning_acronym
python --version # check your python version
pip install -r requirements.txt
```

Finally, fill the config file with appropriate open web ui token and url at [conf/conf.yaml](conf/conf.yaml).

```yaml
owui_conf:
  token: your-token
  url: https://ragarenn.eskemm-numerique.fr/your-mail/api/chat/completions
  fav_model_name: mistralai/Mistral-Small-3.1-24B-Instruct-2503
```
## Source

Training notebook and scripts were adapted from https://colab.research.google.com/drive/1DqKNPOzyMUXmJiJFvJITOahVDxCrA-wA#scrollTo=9Ixtdtpgyv_a; and hugging face documentation (p.e. https://huggingface.co/learn/llm-course/en/chapter11/3).