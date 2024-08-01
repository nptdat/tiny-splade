from typing import Dict

import torch
import torch.nn.functional as F


class DistillMarginMSELoss:
    loss_type = "margin_mse"

    def __init__(self) -> None:
        self.loss = torch.nn.MSELoss()

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """_summary_

        Args:
            inputs (Dict[str, torch.Tensor]):
                Inputs (outputs from model) to compute MarginMSE loss.
                Keys:
                    pos_score: (bs, 1)
                    neg_score: (bs, 1)
                    teacher_pos_score: (bs, 1)
                    teacher_neg_score: (bs, 1)

        Returns:
            torch.Tensor: Loss value (scalar)
        """
        return self.loss(
            inputs["pos_score"] - inputs["neg_score"],
            inputs["teacher_pos_score"] - inputs["teacher_neg_score"],
        )


class DistillKLDivLoss:
    loss_type = "kldiv"

    def __init__(self) -> None:
        self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """_summary_

        Args:
            inputs (Dict[str, torch.Tensor]):
                Inputs (outputs from model) to compute KL-Divergence loss.
                Keys:
                    pos_score: (bs, 1)
                    neg_score: (bs, 1)
                    teacher_pos_score: (bs, 1)
                    teacher_neg_score: (bs, 1)
        Returns:
            torch.Tensor: Loss value (scalar)
        """
        student_scores = torch.hstack(
            [inputs["pos_score"], inputs["neg_score"]]
        )  # (bs, 2)
        teacher_scores = torch.hstack(
            [
                inputs["teacher_pos_score"].unsqueeze(dim=-1),
                inputs["teacher_neg_score"].unsqueeze(dim=-1),
            ]
        )  # (bs, 2)

        inputs_ = F.log_softmax(student_scores, dim=1)  # (bs, 2)
        target = F.log_softmax(teacher_scores, dim=1)  # (bs, 2)
        return self.loss(inputs_, target)
