from pathlib import Path
from typing import Any, List, Set, Type

from tiny_splade.schemas.data import (
    BaseMasterSchema,
    DocumentMasterSchema,
    QueryMasterSchema,
)

from .ndjson_loader import NdjsonLoader

# from transformers import AutoTokenizer


class BaseMaster:
    SCHEMA_CLASS: Type[BaseMasterSchema]

    def __init__(
        self,
        data_path: Path,
        # tokenizer_path: str,
        # max_length: int = 512
    ) -> None:
        self.data_path = data_path
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     tokenizer_path, trust_remote_code=True
        # )
        # self.max_length = max_length

        self.load_data(data_path)

    # def _tokenize_text(
    #     self, text_list: List[str]
    # ) -> Tuple[List[List[int]], List[List[int]]]:
    #     results = self.tokenizer(
    #         text_list,
    #         add_special_tokens=True,
    #         padding="longest",  # pad to max sequence length in batch
    #         truncation="longest_first",  # truncates to self.max_length
    #         max_length=self.max_length,
    #         return_attention_mask=True
    #     )
    #     print(results)
    #     token_ids = results["input_ids"]
    #     attention_mask = results["attention_mask"]
    #     return token_ids, attention_mask

    def load_data(self, data_path: Path) -> None:
        loader = NdjsonLoader(data_path)
        id_list = []
        text_list = []
        for item in loader():
            try:
                data_obj = self.SCHEMA_CLASS(**item)
                id_list.append(data_obj.id)
                text_list.append(data_obj.text)
            except Exception as e:
                raise e

        # TODO: tokenize text
        # token_ids, attention_mask = self._tokenize_text(text_list)
        # self.token_ids = token_ids
        # self.attention_mask = attention_mask

        self.data = {
            id_: dict(text=text) for id_, text in zip(id_list, text_list)
        }
        # QUESTION: Why id_list is necessary?
        self.id_list = id_list
        # for fast existence checking
        self.id_set = set(id_list)

    def get_id_list(self) -> List[int]:
        return self.id_list

    def get_id_set(self) -> Set[int]:
        return self.id_set

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, id: int) -> Any:
        if id not in self.data:
            raise IndexError(f"The id {id} does not exist in the master data")
        return (self.data[id]["text"],)

    def __contains__(self, id: int) -> bool:
        return id in self.id_set


class QueryMaster(BaseMaster):
    SCHEMA_CLASS = QueryMasterSchema


class DocumentMaster(BaseMaster):
    SCHEMA_CLASS = DocumentMasterSchema
