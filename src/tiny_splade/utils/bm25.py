from typing import Callable, List, Optional, Union

import numpy as np
from scipy import sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer


class BM25:
    """
    Implement vanilla Okapi BM25 as in https://en.wikipedia.org/wiki/Okapi_BM25
    """

    def __init__(
        self,
        k1: float = 1.6,
        b: float = 0.75,
        max_features: Optional[int] = None,
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        tokenizer: Optional[Callable] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.tokenizer = tokenizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            norm=None,
            smooth_idf=False,
            tokenizer=tokenizer,
        )

    def _recompute_idf(self, sklearn_idf: np.ndarray) -> np.ndarray:
        """
        Recompute idf for BM25 from scikit-learn TfidfVectorizer.idf_
        NOTE:
            - In TfidfVectorizer, idf(t) = log [ n / df(t) ] + 1 (if smooth_idf=False)
            - In BM25, idf(t) = ln[(n - df(t) + 0.5) / (df(t) + 0.5) + 1]

        sklearn_idf: TfidfVectorizer.idf_, ndarray, shape=(D, 1)
        """
        D = self.D
        df = D / np.exp(sklearn_idf - 1)
        idf: np.ndarray = np.log(((D - df + 0.5) / (df + 0.5)) + 1)
        return idf

    def fit(self, corpus: List[str]) -> "BM15":  # type: ignore
        k1 = self.k1
        b = self.b

        self.vectorizer.fit(corpus)

        # get df via CountVectorizer.transform, the super class of TfidfVectorizer
        tf = super(TfidfVectorizer, self.vectorizer).transform(corpus)  # type: ignore
        doc_lens = tf.sum(axis=1)

        lens_D = doc_lens.A1
        avgdl = doc_lens.mean()
        denom_partial = k1 * (1 - b + b * lens_D / avgdl)

        self.D = len(corpus)  # num of docs in the corpus
        self.V = len(self.vectorizer.vocabulary_)  # size of the vocab
        self.lens_D = lens_D  # lens of each doc
        self.avgdl = avgdl  # average len of docs in the corpus
        self.tf = tf  # spmatrix_csr, shape=(D, V)
        self.denom_partial = denom_partial.reshape(1, -1)  # array, shape=(1, D)
        self.idf = self._recompute_idf(self.vectorizer.idf_)
        return self

    def transform(self, query: str) -> np.ndarray:
        q = super(TfidfVectorizer, self.vectorizer).transform([query])  # type: ignore
        n = len(q.indices)  # num of terms in the query
        k1 = self.k1
        D = self.D
        shape = (n, D)

        tf_ = sps.csc_array(self.tf[:, q.indices].T)  # csc_array, shape=(n, D)
        idf = self.idf[None, q.indices].T  # ndarray, shape=(n, 1)

        numerator = (
            np.broadcast_to(idf, shape) * tf_ * (k1 + 1)
        )  # csr_array, shape=(n, D)
        denominator = tf_ + self.denom_partial  # ndarray, shape=(n, D)
        scores: np.ndarray = (numerator / denominator).sum(
            axis=0
        )  # ndarray, shape=(D, )

        return scores

    def rank_documents(self, query: str) -> np.ndarray:
        scores = self.transform(query)
        return np.argsort(scores)[::-1]
