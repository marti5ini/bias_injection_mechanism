import os
import networkx as nx
from data_generation import CausalDataGenerator
from data_generator_updated import CausalDataGenerator2
from graphviz import Digraph
from parameters import noise, y_binary, naming_structures

def construct_causal_structure(graph, name, i):
    if name == 'mediators':
        if i == 0:
            graph.add_edges_from([('A', 'M1'), ('M1', 'Y')])
            naming_structures[name[:-1] + '_' + str(i)] = 'M11'

        elif i == 1:
            graph.add_edges_from([('A', 'M1'), ('M1', 'Y'),
                                  ('A', 'M2'), ('M2', 'Y'),
                                  ])
            naming_structures[name[:-1] + '_' + str(i)] = 'M22'
        elif i == 2:
            graph.add_edges_from([('A', 'M1'), ('M1', 'M2'), ('M2', 'Y'),
                                  ])
            naming_structures[name[:-1] + '_' + str(i)] = 'M21'
        elif i == 3:
            graph.add_edges_from([('A', 'M1'), ('M1', 'M2'), ('M2', 'Y'),
                                  ('A', 'M3'), ('M3', 'Y'),
                                  ])
            naming_structures[name[:-1] + '_' + str(i)] = 'M32'
        elif i == 4:
            graph.add_edges_from([('A', 'M1'), ('M1', 'M2'), ('M2', 'Y'),
                                  ('A', 'M3'), ('M3', 'M4'),
                                  ('M4', 'Y'),
                                  ])
        elif i == 5:
            graph.add_edges_from([('A', 'M1'), ('M1', 'M2'), ('M2', 'Y'),
                                  ('A', 'M3'), ('M3', 'M4'),
                                  ('M4', 'Y'), ('A', 'M5'), ('M5', 'Y')
                                  ])
        elif i == 6:
            graph.add_edges_from([('A', 'M1'), ('M1', 'M2'), ('M2', 'Y'),
                                  ('M1', 'M4'), ('M1', 'M5'),
                                  ('A', 'M3'), ('M3', 'M4'), ('M4', 'Y'),
                                  ('A', 'M5'), ('M5', 'M6'),
                                  ('M6', 'Y'),
                                  ])
        else:
            graph.add_edges_from([('A', 'M1'), ('M1', 'M2'), ('M2', 'Y'), ('M2', 'M6'),
                                  ('A', 'M3'), ('M3', 'M4'), ('M4', 'Y'), ('A', 'M5'),
                                  ('M5', 'M6'), ('M6', 'M4'), ('M3', 'M7'),
                                  ('M6', 'Y'), ('A', 'M7'), ('M7', 'Y')
                                  ])
    elif name == 'mediators and confounders':
        if i == 0:
            graph.add_edges_from([('A', 'M1'), ('M1', 'Y'), ('C1', 'A'), ('C1', 'Y')])
            naming_structures['med_confounder' + '_' + str(i)] = 'M11 C11'
        elif i == 1:
            graph.add_edges_from([('A', 'M1'), ('M1', 'Y'),
                                  ('C2', 'A'), ('C2', 'Y'),
                                  ('C1', 'A'), ('C1', 'Y'),
                                  ])
            naming_structures['med_confounder' + '_' + str(i)] = 'M11 C22'
        elif i == 2:
            graph.add_edges_from([('A', 'M1'), ('M1', 'Y'),
                                  ('C1', 'A'), ('C1', 'M1'),
                                  ])
            naming_structures['med_confounder' + '_' + str(i)] = 'M11 C21'
        elif i == 3:
            graph.add_edges_from([('A', 'M1'), ('M1', 'Y'),
                                  ('A', 'M2'), ('M2', 'Y'),
                                  ('C1', 'A'), ('C1', 'M1'),
                                  ])
            naming_structures['med_confounder' + '_' + str(i)] = 'M21 C11'
    elif name == 'confounders':
        if i == 0:
            graph.add_edges_from([('C1', 'A'), ('C1', 'Y')])
            naming_structures[name[:-1] + '_' + str(i)] = 'C11'
        else:
            graph.add_edges_from([('C1', 'A'), ('C1', 'Y'),
                                  ('C2', 'A'), ('C2', 'Y'),
                                  ])
            naming_structures[name[:-1] + '_' + str(i)] = 'C22'
    else:
        if i == 0:
            graph.add_edges_from([('A', 'W1'), ('Y', 'W1')])
            naming_structures[name[:-1] + '_' + str(i)] = 'W11'
        else:
            graph.add_edges_from([('A', 'W1'), ('Y', 'W1'), ('A', 'W2'), ('Y', 'W2')])
            naming_structures[name[:-1] + '_' + str(i)] = 'W22'
    return graph


def save_graph(graph, path, structure, name):
    dot = Digraph(format='png')

    for edge in graph.edges:
        dot.edge(edge[0], edge[1])

    ground_truth_path = os.path.join(path, 'ground_truth')
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)
    dot.render(filename=name, directory=os.path.join(ground_truth_path), cleanup=True)
    return


def save_df(data, path, structure, bias):
    data_path = os.path.join(path, 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data.to_csv(os.path.join(data_path, f'{structure}_{bias}.csv'), index=False)
    return

def compute_spd(df):
    from aif360.datasets.standard_dataset import StandardDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    dataset = StandardDataset(df,
                              label_name='Y',
                              favorable_classes=[1],
                              protected_attribute_names=['A'],
                              privileged_classes=[[1]])

    attr = dataset.protected_attribute_names[0]

    idx = dataset.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset.unprivileged_protected_attributes[idx][0]}]

    metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
    return metric.statistical_parity_difference()


def generate_data(structure, structure_type, bias, sensitive_attributes=['A'],
                  A_Y=False, parameter1=1, n_sample=1000, path=None, spd=True, y_binary=y_binary):
    graph = nx.DiGraph()
    parts = structure.split('_')
    i = int(parts[-1])
    graph = construct_causal_structure(graph, structure_type, i)

    if A_Y:
        graph.add_edge('A', 'Y')

    save_graph(graph, path, structure_type, structure)

    model = CausalDataGenerator(
        graph,
        sensitive_attributes=sensitive_attributes,
        parameter1=parameter1,
        parameter2=bias,
        y_binary=y_binary,
        sem_type=noise,
        model_type='linear'
    )
    data = model.generate_data(n_samples=n_sample)

    if spd:
        print(f'SPD for {structure} with param1={parameter1}, param2={bias}:  {round(compute_spd(data), 3)}')

    save_df(data, path, structure, bias)

    return data, graph


def generate_data_new(structure, structure_type, bias, sensitive_attributes='A',
                  A_Y=False, parameter1=1, n_sample=1000, path=None, spd=True, y_binary=y_binary):
    graph = nx.DiGraph()
    parts = structure.split('_')
    i = int(parts[-1])
    graph = construct_causal_structure(graph, structure_type, i)

    if A_Y:
        graph.add_edge('A', 'Y')

    save_graph(graph, path, structure_type, structure)

    model = CausalDataGenerator2(
        graph,
        parameter1=parameter1,
        parameter2=bias,
        sem_type=noise,
    )
    data = model.generate_data(n_samples=n_sample)

    if spd:
        print(f'SPD for {structure} with param1={parameter1}, param2={bias}:  {round(compute_spd(data), 3)}')

    save_df(data, path, structure, bias)

    return data, graph