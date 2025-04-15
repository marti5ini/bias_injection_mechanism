import warnings
from collections import Counter

import numpy as np

from causal_structures import generate_data_new, generate_data
from parameters import *
from plot_utils import *

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    current_directory = os.getcwd()
    print(f'Generation for A_Y={A_Y}')
    if A_Y:
        path = os.path.join(current_directory, 'data', folder_name, 'a_y')
    else:
        path = os.path.join(current_directory, 'data', folder_name, 'no_a_y')

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.isdir(path + '/plots/' + cd_algo):
        os.makedirs(path + '/plots/' + cd_algo)

    print(f'Algorithm: {cd_algo}')
    for i, (type_st, structure) in enumerate(combined.items()):
        print(f'Structure: {type_st}')
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        changes = dict()
        for j, name in enumerate(structure):

            changes[name] = dict()
            for k, parameter2 in enumerate(parameters2_list):
                edges_list = []  # to store edges for each run
                for _ in range(10):
                    df, ground_truth = generate_data(structure=name, structure_type=type_st,
                                       bias=parameter2, sensitive_attributes=['A'], A_Y=A_Y,
                                       parameter1=parameter1, n_sample=n_sample, path=path, spd=False)

                    file_path = (path + '/estimated_graph/' + cd_algo + '/'
                                 + name + '_' + str(parameter2) + '.txt')
                    causal_matrix = np.loadtxt(file_path)
                    estimated_edges = set(map(tuple, np.argwhere(causal_matrix)))
                    column_names = df.columns.tolist()
                    edges_list.append({f'{column_names[edge[0]]}->{column_names[edge[1]]}' for edge in estimated_edges})

                # Calculate the mean of edges over 10 run
                set_counts = Counter(map(frozenset, edges_list))
                # Find the most common set
                most_common_set = set_counts.most_common(1)[0][0]
                changes[name][parameter2] = most_common_set
            dot = plot_using_graphviz(changes[name], parameters2_list, name, axs[j], ground_truth)

            output_path = f'{path}/plots/{cd_algo}/{name}'
            dot.render(output_path, format='pdf', cleanup=True)

        # Create a shared colorbar for all subplots within the figure
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=len(parameters2_list) - 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, ticks=range(len(parameters2_list)))
        cbar.set_ticklabels([str(level) for level in parameters2_list], fontsize=12)
        cbar.set_label('Bias Level', fontsize=12)
        # plt.show()


        file_path = f'{path}/plots/{cd_algo}/{type_st}_dags.pdf'

        if os.path.isfile(file_path):
            os.remove(file_path)

        plt.savefig(file_path)
        plt.close()