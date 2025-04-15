from castle.common import BaseLearner, Tensor
from castle.algorithms.ges.operators import search
from castle.algorithms.ges.score.local_scores import (BICScore, BDeuScore, DecomposableScore)
import numpy as np


class GES_algorithm(BaseLearner):
    """
    Greedy equivalence search for causal discovering

    References
    ----------
    [1]: https://www.sciencedirect.com/science/article/pii/S0888613X12001636
    [2]: https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf

    Parameters
    ----------
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

        Notes:
            1. 'bdeu' just for discrete variable.
            2. if you want to customize criterion, you must create a class
            and inherit the base class `DecomposableScore` in module
            `ges.score.local_scores`
    method: str
        effective when `criterion='bic'`, one of ['r2', 'scatter'].
    k: float, default: 0.001
        structure prior, effective when `criterion='bdeu'`.
    N: int, default: 10
        prior equivalent sample size, effective when `criterion='bdeu'`
    """

    def __init__(self, criterion='bic', method='scatter', k=0.001, N=10):
        super(GES_algorithm, self).__init__()
        if isinstance(criterion, str):
            if criterion not in ['bic', 'bdeu']:
                raise ValueError(f"if criterion is str, it must be one of "
                                 f"['bic', 'bdeu'], but got {criterion}.")
        else:
            if not isinstance(criterion, DecomposableScore):
                raise TypeError(f"The criterion is not instance of "
                                f"DecomposableScore.")
        self.criterion = criterion
        self.method = method
        self.k = k
        self.N = N

    def learn(self, data, columns=None, **kwargs):

        d = data.shape[1]
        e = np.zeros((d, d), dtype=int)

        if self.criterion == 'bic':
            self.criterion = BICScore(data=data,
                                      method=self.method)
        elif self.criterion == 'bdeu':
            self.criterion = BDeuScore(data=data, k=self.k, N=self.N)

        c = search.fes(C=e, criterion=self.criterion)
        c = search.bes(C=c, criterion=self.criterion)

        self._causal_matrix = Tensor(c, index=columns, columns=columns)