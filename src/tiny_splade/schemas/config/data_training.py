from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    training_data_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "data format (json, pkl_dict, saved_pkl, trec, triplets)"
        },
    )
    training_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to training file"}
    )
