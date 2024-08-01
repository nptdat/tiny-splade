import torch


def dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scores = torch.sum(x * y, dim=1, keepdim=True)
    return scores
