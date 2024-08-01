from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BaseMasterSchema(ABC):
    text: str

    @property
    @abstractmethod
    def id(self) -> int:
        raise NotImplementedError("Property id is not implemented yet!")


@dataclass
class QueryMasterSchema(BaseMasterSchema):
    qid: int

    @property
    def id(self) -> int:
        return self.qid


@dataclass
class DocumentMasterSchema(BaseMasterSchema):
    did: int

    @property
    def id(self) -> int:
        return self.did


@dataclass
class PositiveListSchema:
    qid: int
    positive_dids: List[int]


@dataclass
class HardNegativeScoreSchema:
    qid: int
    scores: Dict[int, float]
