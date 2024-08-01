from dataclasses import dataclass, field


@dataclass
class SpladeIndexingArguments:
    indexer_path: str = field(metadata={"help": "Path to indexer"})
