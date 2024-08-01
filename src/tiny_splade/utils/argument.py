import dataclasses
import re
from typing import Any, Type

from omegaconf import DictConfig

RE_PATT = re.compile("<class '(.*)'>")
PRIMITIVE_TYPES = {"bool", "str", "int", "float", "list", "dict"}


def instantiate(data_class: Type, dict_config: DictConfig) -> Any:
    """Instantiate a config object from hydra's DictConfig to avoid
    hydra's `ConfigValueError: Unions of containers are not supported:` error.

    Args:
        data_class (Type): Class of the config object
        dict_config (DictConfig): Config data from hydra in form of DictConfig

    Returns:
        config: The return config object whose type is `data_class`
    """
    fields = dataclasses.fields(data_class)
    params = dict()
    for field in fields:
        if is_primitive(field.type):
            if field.name in dict_config:
                params[field.name] = dict_config[field.name]
        else:
            params[field.name] = instantiate(
                field.type, dict_config[field.name]
            )
    return data_class(**params)


def is_primitive(type_: Any) -> bool:
    """
    Some patterns of type_
        1. <class 'tiny_splade.schemas.args.training.SpladeTrainingArguments'>
        2. <class 'str'>
        3. <class 'list'>
        4. <class 'dict'>
        5. 'typing.Optional[str]'
    The types of pattern 2 ~ 5 are primitive.
    """
    type_str = str(type_)
    if type_str.startswith("<class"):
        matches = RE_PATT.search(type_str)
        if matches:
            type_name = matches.group(1)
            if type_name not in PRIMITIVE_TYPES:
                return False
    return True
