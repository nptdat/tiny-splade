import dataclasses
import re
from dataclasses import dataclass, field  # , MISSING
from typing import List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

# from hydra.utils import instantiate
from pydantic import BaseModel

from tiny_splade.schemas.config import (
    Config,
    DataTrainingArguments,
    ModelArguments,
    SpladeTrainingArguments,
)
from tiny_splade.utils.argument import instantiate


@hydra.main(
    version_base="1.2", config_path="../../config", config_name="splade"
)
def train(cfg: DictConfig) -> None:
    print(cfg)

    # print(OmegaConf.to_yaml(cfg))
    # config = OmegaConf.to_object(cfg)
    # print(config)
    # breakpoint()
    config = instantiate(Config, cfg)
    # breakpoint()
    print(config.data)
    print(f"{config.training.learning_rate=}")
    print(f"{config.training.regularizers=}")
    print(f"{config.indexing=}")
    print(f"{config.indexing.indexer_path=}")


if __name__ == "__main__":
    train()
