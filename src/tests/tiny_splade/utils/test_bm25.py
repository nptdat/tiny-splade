from typing import List

import numpy as np
import pytest
from numpy import testing as np_testing

from tiny_splade.utils import BM25

corpus = [
    "There is a good pianist",
    "It is quite windy in London today",
    "He is playing piano in London and he is the best there",
]


class TestBM25:
    @pytest.mark.parametrize(
        """corpus, exp_avgdl, exp_V, exp_D, exp_lens_D, exp_denom_partial, """
        """exp_idf""",
        [
            (
                corpus,
                7.666666666666667,
                16,
                3,
                np.array([4, 7, 12]),
                np.array([[1.02608696, 1.49565217, 2.27826087]]),
                np.array(
                    [
                        0.98082925,
                        0.98082925,
                        0.98082925,
                        0.98082925,
                        0.47000363,
                        0.13353139,
                        0.98082925,
                        0.47000363,
                        0.98082925,
                        0.98082925,
                        0.98082925,
                        0.98082925,
                        0.98082925,
                        0.47000363,
                        0.98082925,
                        0.98082925,
                    ]
                ),
            )
        ],
    )
    def test_fit(
        self,
        corpus: List[str],
        exp_avgdl: float,
        exp_V: int,
        exp_D: int,
        exp_lens_D: np.ndarray,
        exp_denom_partial: np.ndarray,
        exp_idf: np.ndarray,
    ) -> None:
        bm25 = BM25()
        bm25.fit(corpus)

        np_testing.assert_almost_equal(bm25.lens_D, exp_lens_D)
        assert bm25.avgdl == exp_avgdl
        assert bm25.V == exp_V
        assert bm25.D == exp_D
        np_testing.assert_almost_equal(bm25.denom_partial, exp_denom_partial)
        np_testing.assert_almost_equal(bm25.idf, exp_idf)

    @pytest.mark.parametrize(
        "query, exp_scores",
        [("windy London", np.array([0.0, 1.51149488, 0.3727615]))],
    )
    def test_transform(self, query: str, exp_scores: np.ndarray) -> None:
        bm25 = BM25()
        bm25.fit(corpus)
        scores = bm25.transform(query)
        np_testing.assert_almost_equal(scores, exp_scores)

    @pytest.mark.parametrize(
        "query, exp_ranks", [("windy London", np.array([1, 2, 0]))]
    )
    def test_rank_documents(self, query: str, exp_ranks: np.ndarray) -> None:
        bm25 = BM25()
        bm25.fit(corpus)
        ranks = bm25.rank_documents(query)
        np_testing.assert_almost_equal(ranks, exp_ranks)
