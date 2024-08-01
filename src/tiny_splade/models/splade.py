"""
References:
[1] From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models
    More Effective.
    - Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane
      Clinchant.
    - SIGIR22 short paper (extension of SPLADE v2) (v2bis, SPLADE++)
    - https://arxiv.org/pdf/2205.04733
"""

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpladeEncoder(torch.nn.Module):
    def __init__(self, model_path: Path):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outs.logits  # (b, N, V)

        # Eq. (1) in [1]
        vec, indices_ = torch.max(
            torch.log(1 + torch.relu(logits))
            * attention_mask.unsqueeze(-1),  # (b, N, 1)
            dim=1,
        )
        return (
            vec  # (b, V), weights `w_j` of the input query/doc over the vocab
        )

    def to_sparse(self, dense: torch.Tensor) -> dict[str, float]:
        """_summary_

        Args:
            dense (torch.Tensor): (b, V)

        Returns:
            dict[str, float]: mapping from word to score that represent a sparse vector
        """
        # TODO: check this code to support batch (this code currently supports only 1 vector)
        # extract non-zero positions
        cols = dense.nonzero().squeeze().cpu().tolist()
        if not isinstance(cols, list):
            cols = [cols]
        print(f"Num of non-zero values: {len(cols)}")

        # extract the non-zero values
        weights = dense[cols].cpu().tolist()
        # use to create a dictionary of token ID to weight
        # sparse_dict = dict(zip(cols, weights))

        # extract the ID position to text token mappings
        idx2token = {
            idx: token for token, idx in self.tokenizer.get_vocab().items()
        }

        # map token IDs to human-readable tokens
        sparse_dict_tokens_: dict[str, float] = {
            idx2token[idx]: round(weight, 2)
            for idx, weight in zip(cols, weights)
        }
        # sort so we can see most relevant tokens first
        sparse_dict_tokens: dict[str, float] = {
            k: v
            for k, v in sorted(
                sparse_dict_tokens_.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        return sparse_dict_tokens


class Splade(torch.nn.Module):
    def __init__(
        self,
        d_model_path: Path,
        q_model_path: Optional[Path] = None,
    ) -> None:
        super().__init__()

        self.d_encoder = SpladeEncoder(d_model_path)
        self.q_encoder = self.d_encoder
        if q_model_path:
            self.q_encoder = SpladeEncoder(q_model_path)

    def forward(
        self,
        q_input_ids: Optional[torch.Tensor] = None,
        q_attention_mask: Optional[torch.Tensor] = None,
        d_input_ids: Optional[torch.Tensor] = None,
        d_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        # print(f"{q_attention_mask=}")
        q_vec = None
        if q_input_ids is not None and q_attention_mask is not None:
            q_vec = self.q_encoder(
                input_ids=q_input_ids, attention_mask=q_attention_mask
            )

        d_vec = None
        if d_input_ids is not None and d_attention_mask is not None:
            d_vec = self.d_encoder(
                input_ids=d_input_ids, attention_mask=d_attention_mask
            )

        return dict(q_vector=q_vec, d_vector=d_vec)

    def to_sparse(
        self, denses: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, float]]:
        sparses = dict()
        if "q_vector" in denses:
            sparses["q_vector"] = self.q_encoder.to_sparse(denses["q_vector"])
        if "d_vector" in denses:
            sparses["d_vector"] = self.d_encoder.to_sparse(denses["d_vector"])
        return sparses
