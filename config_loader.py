from pydantic import (
    BaseModel,
    PositiveInt,
    FilePath,
    DirectoryPath,
    PositiveFloat,
    HttpUrl,
    SecretStr,
)
from typing import Literal
import json
import yaml


from transformers.trainer_utils import SchedulerType


def pydantic_parser(txt):
    """
    Parse the config file with pydantic.
    """

    class ConfigSchema(BaseModel):
        infra: Literal["onyxia", "local", "datalab_gcp"]
        resume_from_checkpoint: DirectoryPath | bool
        model_name: str
        model_dtype: str
        train_dataset: FilePath
        data_prop: PositiveFloat
        test_dataset: FilePath
        output_dir: DirectoryPath
        training_params: TrainingParamScheme
        max_new_tokens: PositiveInt
        owui_conf: OWUIConfScheme
        device: str
        mlflow_uri: str

    cfg = yaml.safe_load(txt)

    ConfigSchema.model_validate(cfg)
    return cfg


class OWUIConfScheme(BaseModel):
    token: SecretStr
    url: HttpUrl
    fav_model_name: str


class LoraParamScheme(BaseModel):
    alpha: PositiveFloat
    rank: PositiveInt
    dropout: PositiveFloat


class TrainingParamScheme(BaseModel):
    learning_rate: PositiveFloat
    learning_rate_scheduler: SchedulerType
    n_epochs: PositiveInt
    logging_steps: PositiveInt
    lora: LoraParamScheme
