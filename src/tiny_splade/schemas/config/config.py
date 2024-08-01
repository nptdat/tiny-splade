from dataclasses import dataclass

from .data_training import DataTrainingArguments
from .indexing import SpladeIndexingArguments
from .model import ModelArguments
from .training import SpladeTrainingArguments


@dataclass
class Config:
    data: DataTrainingArguments
    model: ModelArguments
    training: SpladeTrainingArguments
    indexing: SpladeIndexingArguments
