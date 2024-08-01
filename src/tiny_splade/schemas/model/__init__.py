from dataclasses import dataclass, field

import torch


@dataclass
class TripletWithDistillationBatch:
    q_input_ids: torch.Tensor
    q_attention_mask: torch.Tensor
    pos_input_ids: torch.Tensor
    pos_attention_mask: torch.Tensor
    neg_input_ids: torch.Tensor
    neg_attention_mask: torch.Tensor
    teacher_pos_scores: torch.Tensor
    teacher_neg_scores: torch.Tensor
