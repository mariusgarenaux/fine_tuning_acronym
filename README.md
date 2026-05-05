# Fine Tuning LLM

This projects aims at fine-tuning a LLM in order to make it understand and memorize a list of given acronyms and their definitions.

We use the frameworks [mlflow](https://mlflow.org/) and [metaflow](https://metaflow.org/). _Metaflow_ as a workflow management system, and _mlflow_ to keep track of previous runs.

The workflow uses the hugging face [transformer](https://huggingface.co/docs/transformers/index) library to load and train a LLM. Then the model is tested using [sentence-transformer](https://www.sbert.net/) library for output similarities, as well a LLM as a judge (called _via_ an OpenWebUI API - here [RAGaRenn](https://ragarenn.eskemm-numerique.fr/index.html)).

You can access a smaller (~ tutorial) version of this project by going in the branch _formation-continue_.

## Project structure

- [fta/main.py](fta/main.py) contains the workflow, (see [metaflow](https://metaflow.org/) documentation). It can be started with : `python mlflow.py run`.

- [fta/programmer.py](fta/programmer.py) file to make multiple runs of the workflow, with different _configurations_. It can be adapted easily.

- [fta/fta_config_loader.py](fta/fta_config_loader.py) loads the config file with pydantic checks, for the workflow.

- [fta/fta_tools.py](fta/fta_tools.py) contains diverses tools used in the workflow, like an OpenWebUIConnector to make authenticated calls to the OpenWebUI API.

## Data

You'll see example data in the folder [example_data](example_data). The training data (as well as test one) contains conversations about acronyms and their definitions. You can use any dataset with the same structure. If you want to generate data from a list acronym, you can take a look on the simplified version of this project, on the branch _formation-continue_ : [https://github.com/mariusgarenaux/fine_tuning_acronym/tree/formation-continue](https://github.com/mariusgarenaux/fine_tuning_acronym/tree/formation-continue).

## Getting started

Git clone the project, set up a python .venv, activate and install libraries :

> Supported with Python 3.13 - not tested with earlier versions (might cause problems with recent ML libs - transformers, ...)

```bash
git clone https://github.com/mariusgarenaux/fine_tuning_acronym
cd fine_tuning_acronym
pip install -r requirements.txt # optionnaly, run uv sync
```

Finally, fill the config file [conf/conf.yaml](conf/conf.yaml). All can be overriden with command line args, thanks to metaflow. Here is an example :

```yaml
infra: local
resume_from_checkpoint: False
model_name: meta-llama/Llama-3.2-1B-Instruct
model_dtype: bfloat16
train_dataset: bucket/data/train_dataset.json
test_dataset: bucket/data/eval_dataset.json
data_prop: 1
output_dir: output
training_params:
  learning_rate: 1e-4
  n_epochs: 4
  learning_rate_scheduler: constant
  logging_steps: 50
  lora:
    alpha: 1
    rank: 16
    dropout: .1
max_new_tokens: 100
device: auto
mlflow_uri: http://127.0.0.1:5000
owui_conf:
  token: abcde
  url: https://ragarenn.eskemm-numerique.fr/name/api/chat/completions
  fav_model_name: mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

To run the workflow, you can (optional) first run the mlflow server : `mlflow server`. Then run `python fta/main.py run` to run the metaflow workflow. If the mlflow server is running (optional), you can follow the metrics in live [http://127.0.0.1:5000/](http://127.0.0.1:5000/) by finding your run.

### Link to Hugging Face

If you finetune a model that need a HuggingFace authentication, you have to give your HuggingFace token by running in a terminal :

```bash
python -c "from huggingface_hub import login; login()"
```

## Source

Training notebook and scripts were adapted from https://colab.research.google.com/drive/1DqKNPOzyMUXmJiJFvJITOahVDxCrA-wA#scrollTo=9Ixtdtpgyv_a; and hugging face documentation (p.e. https://huggingface.co/learn/llm-course/en/chapter11/3).
