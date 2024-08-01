from pathlib import Path
from typing import Dict, List

from tiny_splade.schemas.data import HardNegativeScoreSchema

from .ndjson_loader import NdjsonLoader


class PairScore:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.load_data(data_path)

    def load_data(self, data_path: Path) -> None:
        loader = NdjsonLoader(data_path)
        # map from `qid` to a dict of scores (did -> score)
        pair_scores: Dict[int, Dict[int, float]] = dict()
        for item in loader():
            obj = HardNegativeScoreSchema(**item)
            # ndjson may save doc_id as str (e.g., '123': 0.75).
            # ensure the doc_ids are integer
            pair_scores[obj.qid] = {
                int(did): score for did, score in obj.scores.items()
            }
        self.pair_scores = pair_scores

    def __getitem__(self, qid: int) -> Dict[int, float]:
        if qid not in self.pair_scores:
            raise IndexError(
                f"The qid {qid} does not exist in hard-negative scores"
            )
        return self.pair_scores[qid]
