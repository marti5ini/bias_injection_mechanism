from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from castle.algorithms import PC
from castle.algorithms.pc.pc import find_skeleton
from castle.common import Tensor
from castle.common.priori_knowledge import orient_by_priori_knowledge

class PC_algorithm(PC):

    def __init__(self, ci_test='fisherz', variant='original'):
        super(PC_algorithm, self).__init__(
            ci_test=ci_test,
            variant=variant
        )

    def learn(self, data, columns=None, use_fifth_rule=True, **kwargs):
        data = Tensor(data, columns=columns)

        skeleton, sep_set = find_skeleton(data,
                                          alpha=self.alpha,
                                          ci_test=self.ci_test,
                                          variant=self.variant,
                                          priori_knowledge=self.priori_knowledge,
                                          **kwargs)

        self._causal_matrix = Tensor(
            orient_update(skeleton, sep_set, use_fifth_rule, self.priori_knowledge).astype(int),
            index=data.columns,
            columns=data.columns,
        )


def orient_update(skeleton, sep_set, use_fifth_rule, priori_knowledge=None):
        """
        Extending the Skeleton to the Equivalence Class

        it orients the undirected edges to form an equivalence class of DAGs.

        Parameters
        ----------
        skeleton : array
            The undirected graph
        sep_set : dict
            separation sets
            if key is (x, y), then value is a set of other variables
            not contains x and y

        Returns
        -------
        out : array
            An equivalence class of DAGs can be uniquely described
            by a completed partially directed acyclic graph (CPDAG)
            which includes both directed and undirected edges.
        """
        if priori_knowledge is not None:
            skeleton = orient_by_priori_knowledge(skeleton, priori_knowledge)

        columns = list(range(skeleton.shape[1]))
        cpdag = deepcopy(abs(skeleton))
        # rule1
        for ij in sep_set.keys():
            i, j = ij
            all_k = [x for x in columns if x not in ij]
            for k in all_k:
                if cpdag[i, k] + cpdag[k, i] != 0 \
                        and cpdag[k, j] + cpdag[j, k] != 0:
                    if k not in sep_set[ij]:
                        if cpdag[i, k] + cpdag[k, i] == 2:
                            cpdag[k, i] = 0
                        if cpdag[j, k] + cpdag[k, j] == 2:
                            cpdag[k, j] = 0
        while True:
            old_cpdag = deepcopy(cpdag)
            pairs = list(combinations(columns, 2))
            for ij in pairs:
                i, j = ij
                if cpdag[i, j] * cpdag[j, i] == 1:
                    # rule2
                    for i, j in permutations(ij, 2):
                        all_k = [x for x in columns if x not in ij]
                        for k in all_k:
                            if cpdag[k, i] == 1 and cpdag[i, k] == 0 \
                                    and cpdag[k, j] + cpdag[j, k] == 0:
                                cpdag[j, i] = 0
                    # rule3
                    for i, j in permutations(ij, 2):
                        all_k = [x for x in columns if x not in ij]
                        for k in all_k:
                            if (cpdag[i, k] == 1 and cpdag[k, i] == 0) \
                                    and (cpdag[k, j] == 1 and cpdag[j, k] == 0):
                                cpdag[j, i] = 0
                    # rule4
                    for i, j in permutations(ij, 2):
                        for kl in sep_set.keys():  # k and l are nonadjacent.
                            k, l = kl
                            # if i——k——>j and  i——l——>j
                            if cpdag[i, k] == 1 \
                                    and cpdag[k, i] == 1 \
                                    and cpdag[k, j] == 1 \
                                    and cpdag[j, k] == 0 \
                                    and cpdag[i, l] == 1 \
                                    and cpdag[l, i] == 1 \
                                    and cpdag[l, j] == 1 \
                                    and cpdag[j, l] == 0:
                                cpdag[j, i] = 0
                    if use_fifth_rule:
                        cpdag = fifth_rule(cpdag, ij, sep_set, columns)
            if np.all(cpdag == old_cpdag):
                break

        return cpdag


def fifth_rule(cpdag, ij, sep_set, columns):
    for i, j in permutations(ij, 2):
        for kj in sep_set.keys():  # k and j are nonadjacent.
            if j not in kj:
                continue
            else:
                kj = list(kj)
                kj.remove(j)
                k = kj[0]
                ls = [x for x in columns if x not in [i, j, k]]
                for l in ls:
                    if cpdag[k, l] == 1 \
                            and cpdag[l, k] == 0 \
                            and cpdag[i, k] == 1 \
                            and cpdag[k, i] == 1 \
                            and cpdag[l, j] == 1 \
                            and cpdag[j, l] == 0:
                        cpdag[j, i] = 0
        return cpdag


def get_causal_matrix(causal_matrix):
    lower = np.tril(causal_matrix)
    upper = np.triu(causal_matrix)
    n, d = lower.shape

    result = np.zeros((n, d), dtype=int)
    for row in range(n):
        for col in range(d):
            if lower[row, col] == 1 and upper[col, row] == 0:
                result[col, row] = 1
            elif lower[row, col] == 0 and upper[col, row] == 1:
                result[col, row] = 1
            elif lower[row, col] == 1 and upper[col, row] == 1:
                result[col, row] = 1
            else:
                pass
    return result

if __name__ == '__main__':

    data = pd.read_csv('/Users/martina/Downloads/synthetic_th_0.0_df_0.csv', index_col=False)

    method = PC_algorithm(ci_test='chi2')
    method.learn(data)
    print(f'Original Version: \n {method.causal_matrix}')
    causal_matrix = get_causal_matrix(method.causal_matrix)
    print(f'\n Our new version: \n {causal_matrix}')

