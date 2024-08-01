from collections import defaultdict
from typing import Any, List, Union

import numpy as np
import torch
from transformers.trainer import Trainer

from tiny_splade.losses import DistillKLDivLoss, DistillMarginMSELoss
from tiny_splade.models import Splade
from tiny_splade.schemas.model import TripletWithDistillationBatch
from tiny_splade.utils.scoring import dot_product


class SpladeTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(SpladeTrainer, self).__init__(*args, **kwargs)
        # the kwargs contains `args: TrainingArguments` param, which is
        # assigned to self by Trainer class

        self._init_losses()
        self._init_regularizers()
        self.score_fn = dot_product  # currently, fixed
        self._reset_states()

    def _init_losses(self) -> None:
        self.losses: List[Union[DistillKLDivLoss, DistillMarginMSELoss]] = []
        if "kldiv" in self.args.training_loss:
            self.losses.append(DistillKLDivLoss())
        if "margin_mse" in self.args.training_loss:
            self.losses.append(DistillMarginMSELoss())

    def _init_regularizers(self) -> None:
        # self.args.regularizers
        pass

    def _reset_states(self) -> None:
        self.accum_loss_values: dict = defaultdict(list)

    def compute_loss(
        self,
        model: Splade,
        inputs: TripletWithDistillationBatch,
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """How the loss is computed by Trainer. By default, all models return
        the loss in the first element.
        Subclass and override for custom behavior.
        """
        # inputs: TripletWithDistillationBatch
        q_pos_vectors: dict = model(
            q_input_ids=inputs.q_input_ids,
            q_attention_mask=inputs.q_attention_mask,
            d_input_ids=inputs.pos_input_ids,
            d_attention_mask=inputs.pos_attention_mask,
        )
        neg_vectors: dict = model(
            q_input_ids=inputs.q_input_ids,
            q_attention_mask=inputs.q_attention_mask,
            d_input_ids=inputs.neg_input_ids,
            d_attention_mask=inputs.neg_attention_mask,
        )
        q_vector = q_pos_vectors["q_vector"]  # (b, V)
        pos_vector = q_pos_vectors["d_vector"]  # (b, V)
        neg_vector = neg_vectors["d_vector"]  # (b, V)
        params = dict(
            pos_score=self.score_fn(q_vector, pos_vector),
            neg_score=self.score_fn(q_vector, neg_vector),
            teacher_pos_score=inputs.teacher_pos_scores,
            teacher_neg_score=inputs.teacher_neg_scores,
        )

        loss_value: torch.Tensor = torch.tensor(0)
        for loss in self.losses:
            loss_value_ = loss(params)
            loss_value += loss_value_
            self.accum_loss_values[loss.loss_type].append(
                loss_value_.cpu().detach().item()
            )
        self.accum_loss_values["sum_loss"].append(
            loss_value.cpu().detach().item()
        )

        # regularizers
        if return_outputs:
            return loss_value, [(q_pos_vectors, neg_vectors)]
        else:
            return loss_value

    def log(self, logs: dict[str, float]) -> None:
        for loss_id, loss_values in self.accum_loss_values.items():
            logs[loss_id] = np.mean(loss_values)
        self._reset()
