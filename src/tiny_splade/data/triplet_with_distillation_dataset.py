import random
from logging import getLogger
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DefaultDataCollator
from transformers.tokenization_utils_base import BatchEncoding

from tiny_splade.schemas.model import TripletWithDistillationBatch

from .master import DocumentMaster, QueryMaster
from .pair_score import PairScore
from .positive_list import PositiveList

logger = getLogger(__name__)


class TripletWithDistillationDataset(Dataset):
    """Dataset for triplets (query, positive, negative) with distillation
    support by augmenting similarity scores for (query, positive) and
    (query, negative) pairs from a teacher model (e.g., a strong
    cross-encoder model).

    One sample of this dataset is a 5-tuple
    - Query text
    - Positive document text
    - Negative document text
    - Similar score between query & positive document from teacher model
    - Similar score between query & negative document from teacher model
    """

    def __init__(
        self,
        query_master_data_path: Path,
        doc_master_data_path: Path,
        positive_pair_data_path: Path,
        hard_negative_scores_data_path: Path,
        sampling_mode: str = "query_based",
        random_seed: Optional[int] = 42,
    ) -> None:
        """Number of samples in the dataset depends on the `sampling_mode` param:
        - `query_based`: (default) in this mode, 1 sample corresponds to 1 query. The
          positive document is sampled from a list of positives of the query.
          Therefore, the number of samples equals to the number of query.
          Training with a single epoch using this mode does not utilize all of
          the positive pairs. So, training with multi-epoch is recommended.
          Currently, only this mode is supported.

        - `positive_pair_based`: in this mode, 1 sample corresponds to a pair
          of (query, positive). Therefore, the number of samples equals to the
          total number of positives of all queries in the dataset.
          Training with only a single epoch ensures that all of the positive
          pairs are accessed.

        Args:
            query_master_data_path (Path): folder or file of query master
            doc_master_data_path (Path): folder or file of document master
            positive_pair_data_path (Path): folder or file of positive pair data
                collection
            hard_negative_scores_data_path (Path): folder or file of
                hard-negative scores
            sampling_mode (str, optional): Indicate how to sample data. Can be
                `query_based` and `positive_pair_based`. Default: `query_based`
            random_seed (int, optional): if random.seed is set outside and
                the callee does want the seed to be set again, set this param
                to None.
        """
        if sampling_mode not in ["query_based", "positive_pair_based"]:
            raise ValueError(
                "`sampling_mode` must be 'query_based' or 'positive_pair_based'"
            )
        if sampling_mode != "query_based":
            raise ValueError("Currently, only 'query_based' is supported")

        logger.info("Loading query master...")
        self.query_master_data_path = query_master_data_path
        self.queries = QueryMaster(query_master_data_path)
        self.qid_list = self.queries.get_id_list()

        logger.info("Loading document master...")
        self.doc_master_data_path = doc_master_data_path
        self.docs = DocumentMaster(doc_master_data_path)

        logger.info("Loading positive lists...")
        self.positive_pair_data_path = positive_pair_data_path
        self.positive_list = PositiveList(positive_pair_data_path)

        logger.info("Loading hard-negative scores...")
        self.hard_negative_scores_data_path = hard_negative_scores_data_path
        self.similarities = PairScore(hard_negative_scores_data_path)

        self.sampling_mode = sampling_mode
        if random_seed:
            random.seed(random_seed)

        self._validate()

    def _validate(self) -> None:
        logger.info("Validating PairsWithDistillDataset...")
        # every qid must appear in query master
        for qid in self.positive_list:
            if qid not in self.queries:
                raise ValueError(
                    f"qid {qid} from positive list does not exist in query master"
                )

        # every did must appear in query master
        for qid in self.positive_list:
            positive_dids = self.positive_list[qid]
            for did in positive_dids:
                if did not in self.docs:
                    raise ValueError(
                        f"did {did} from positive list does not exist in document master"
                    )

        # every query must has at least 1 positive document
        for qid in self.positive_list:
            if len(self.positive_list[qid]) == 0:
                raise ValueError(f"qid {qid} has no positive document")

        # every (qid, positive_did) pair must have teacher_pair_score
        for qid in self.positive_list:
            positive_dids = self.positive_list[qid]
            for did in positive_dids:
                scores = self.similarities[qid]
                if did not in scores:
                    raise ValueError(
                        f"The pair ({qid}, {did}) (qid, did) has not hard-negative score"
                    )

        # every query must have at least 1 available negative document
        for qid in self.positive_list:
            pos_dids = self.positive_list[qid]
            sim_scores = self.similarities[qid]
            neg_dids = set(sim_scores.keys()) - set(pos_dids)
            neg_dids = neg_dids.intersection(self.docs.get_id_set())
            if len(neg_dids) == 0:
                raise ValueError(
                    f"qid {qid} has no available negative document"
                )

    def __len__(self) -> int:
        return len(self.qid_list)

    def __getitem__(self, index: int) -> Tuple[str, str, str, float, float]:
        qid = self.qid_list[index]
        q_text = self.queries[qid][0]
        sim_scores = self.similarities[qid]

        pos_dids = self.positive_list[qid]
        neg_dids_set = set(sim_scores.keys()) - set(pos_dids)
        neg_dids = list(neg_dids_set.intersection(self.docs.get_id_set()))

        pos_did = random.sample(pos_dids, 1)[0]
        neg_did = random.sample(neg_dids, 1)[0]
        pos_text = self.docs[pos_did][0]
        neg_text = self.docs[neg_did][0]
        score_pos = sim_scores[pos_did]
        score_neg = sim_scores[neg_did]

        return q_text, pos_text, neg_text, score_pos, score_neg


class TripletWithDistillationCollator(DefaultDataCollator):
    """Provide collator for the TripletWithDistillationDataset."""

    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        print(self)
        super(TripletWithDistillationCollator, self).__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self.max_length = max_length

    def _tokenize(self, text_list: List[str]) -> BatchEncoding:
        outputs = self.tokenizer(
            text_list,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return outputs

    def collate_fn(self, data: list) -> Any:
        q_text, pos_text, neg_text, score_pos, score_neg = zip(*data)

        q_outputs = self._tokenize(list(q_text))
        pos_outputs = self._tokenize(list(pos_text))
        neg_outputs = self._tokenize(list(neg_text))

        batch = TripletWithDistillationBatch(
            q_input_ids=torch.tensor(q_outputs["input_ids"]),
            q_attention_mask=torch.tensor(q_outputs["attention_mask"]),
            pos_input_ids=torch.tensor(pos_outputs["input_ids"]),
            pos_attention_mask=torch.tensor(pos_outputs["attention_mask"]),
            neg_input_ids=torch.tensor(neg_outputs["input_ids"]),
            neg_attention_mask=torch.tensor(neg_outputs["attention_mask"]),
            teacher_pos_scores=torch.tensor(score_pos),
            teacher_neg_scores=torch.tensor(score_neg),
        )
        return batch
