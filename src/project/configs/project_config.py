import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


def parse_config(config: dict | str) -> ProjectConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    return ProjectConfig(**config)
