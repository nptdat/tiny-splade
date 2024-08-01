import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest
from pytest import FixtureRequest

from tiny_splade.utils.argument import is_primitive


@pytest.fixture
def class_type() -> Any:
    @dataclass
    class Config:
        field_dummy1: str

    @dataclass
    class ClassWithTypes:
        field_str: str
        field_int: int
        field_float: float
        field_list: list
        field_list_int: list[int]
        field_dict: dict
        field_dict_str_int: dict[str, int]
        field_config: Config

    return ClassWithTypes


@pytest.mark.parametrize(
    "type_class_fixture, exp",
    [
        (
            "class_type",
            {
                "field_str": True,
                "field_int": True,
                "field_float": True,
                "field_list": True,
                "field_list_int": True,
                "field_dict": True,
                "field_dict_str_int": True,
                "field_config": False,
            },
        )
    ],
)
def test_is_primitive(
    request: FixtureRequest, type_class_fixture: str, exp: dict
) -> None:
    type_class = request.getfixturevalue(type_class_fixture)
    fields = dataclasses.fields(type_class)
    for field in fields:
        assert is_primitive(field.type) == exp[field.name]
