from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class FLOPSRegularizer:
    reg_type: str
    lambda_: float


@dataclass
class SpladeRegularizers:
    doc: FLOPSRegularizer
    query: FLOPSRegularizer


@dataclass
class SpladeTrainingArguments(TrainingArguments):
    # @dataclass
    # class SpladeTrainingArguments:
    # regularizers: SpladeRegularizers
    regularizers: SpladeRegularizers = field(
        metadata=dict(help="FLOPS style regularizers on `doc` & `query`"),
    )

    training_loss: str = field(
        default="kldiv,margin_mse",
        metadata=dict(
            help="""Training loss function names separated by `,`. """
            """Currently support {`kldiv`, `margin_mse`}"""
        ),
    )
    param2: str = field(default="xyz2", metadata=dict(help="Just for test"))
