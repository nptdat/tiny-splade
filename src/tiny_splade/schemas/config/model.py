from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )

    max_length: int = field(
        default=128, metadata={"help": "Max length for sequences"}
    )
