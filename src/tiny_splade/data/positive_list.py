from collections.abc import Iterator
from pathlib import Path
from typing import Dict, List

from tiny_splade.schemas.data import PositiveListSchema

from .ndjson_loader import NdjsonLoader


class PositiveList:
    """Map from a query ID to list of positive document IDs"""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.load_data(data_path)

    def load_data(self, data_path: Path) -> None:
        loader = NdjsonLoader(data_path)
        # map from `qid` to list of `positive dids`
        positives: Dict[int, List[int]] = dict()
        for item in loader():
            obj = PositiveListSchema(**item)
            positives[obj.qid] = obj.positive_dids
        self.positives = positives

    def __getitem__(self, qid: int) -> List[int]:
        if qid not in self.positives:
            raise IndexError(f"The qid {qid} does not exist in positive list")
        return self.positives[qid]

    def __iter__(self) -> Iterator[int]:
        yield from self.positives.keys()
