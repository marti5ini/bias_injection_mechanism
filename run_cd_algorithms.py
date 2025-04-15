import os

import pandas as pd
import numpy as np

from cd_algorithms.pc import get_causal_matrix, PC_algorithm
from cd_algorithms.ges import GES_algorithm
from cd_algorithms.lingam import DirectLiNGAM_algorithm


def run_causal_discovery(structure, path, parameter2, cd_algorithm):

    if cd_algorithm == 'ges':
        algorithm_name = GES_algorithm()
    elif cd_algorithm == 'pc':
        algorithm_name = PC_algorithm(variant='stable')
    elif cd_algorithm == 'lingam':
        algorithm_name = DirectLiNGAM_algorithm()
    else:
        raise NotImplemented

    data = pd.read_csv(path + '/data/' + structure + '_' + str(parameter2) + '.csv')
    method = algorithm_name
    method.learn(data)
    causal_matrix = get_causal_matrix(method.causal_matrix)

    if not os.path.exists(path + '/estimated_graph/' + cd_algorithm):
        os.makedirs(path + '/estimated_graph/'+ cd_algorithm)

    np.savetxt(path + '/estimated_graph/' + cd_algorithm + '/' + structure + '_'
               + str(parameter2) + '.txt', causal_matrix)

    return causal_matrix
