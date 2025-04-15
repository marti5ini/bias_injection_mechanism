import os

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from parameters import naming_structures
import graphviz
import matplotlib.colors as mcolors
from PIL import Image
import io

cmap = plt.cm.coolwarm


def plot_colored_graph(changes, parameters_list, structure, ax, ground_truth):
    # Initialize a dictionary to store the counts of each edge at different discrimination levels
    edge_counts = {}

    # Iterate over discrimination levels and edges in the changes dictionary
    for discrimination_level, edges in changes.items():
        for edge in edges:
            # Update edge_counts dictionary
            if edge in edge_counts:
                edge_counts[edge].append(discrimination_level)
            else:
                edge_counts[edge] = [discrimination_level]

    # Create a mapping from parameter names to their indices
    parameter_to_index = {parameters_list[i]: i for i in range(len(parameters_list))}

    # Initialize a dictionary to store edges and their associated discrimination levels
    edges_founded = {}

    # Iterate over edges and their discrimination levels
    for edge, levels in edge_counts.items():
        cnt = 0
        temp = None
        # Check if an edge is associated with only one discrimination level
        if len(levels) == 1:
            cnt += 1
            temp = levels[0]
        # Check for edges associated with the last parameter in the list
        for parameter in levels:
            if parameter_to_index[parameter] == len(parameters_list) - 1:
                cnt += 1
                if temp is None:
                    temp = parameter
                break
            # Check if the next parameter in the list is also present in the discrimination levels
            if parameters_list[parameter_to_index[parameter] + 1] in levels:
                cnt += 1
                if temp is None:
                    temp = parameter
            # Break if at least two conditions are met
            if cnt == 2:
                break
        # If at least one condition is met, add the edge to the edges_founded dictionary
        if cnt >= 1:
            edges_founded[edge] = temp

    print(edges_founded)

    # Create a directed graph using NetworkX
    G = nx.DiGraph()

    # Create a list to store unique colors
    unique_colors = []

    # Iterate over edges and their associated discrimination levels
    for edge, weight in edges_founded.items():
        source, target = edge.split('->')
        unique_colors.append((source, target))
        G.add_edge(source, target, weight=weight)

    # Custom layout using the spring_layout algorithm
    pos = nx.spring_layout(G, seed=65, k=1500, iterations=300)

    # Normalize weights for color mapping
    norm = Normalize(vmin=min(parameters_list), vmax=max(parameters_list))

    # Create a list of edge colors based on the normalized weights
    edge_colors = [cmap(norm(G.edges[edge]['weight'])) for edge in G.edges]

    # Create a list of edge styles (solid or dashed)
    edge_styles = []
    ground_truth_edges = []
    for edge in G.edges:
        if ground_truth.has_edge(edge[0], edge[1]):
            edge_styles.append('solid')
            ground_truth_edges.append((edge[0], edge[1]))
        else:
            edge_styles.append('dashed')
    for edge in ground_truth.edges:
        if (edge[0], edge[1]) not in ground_truth_edges:
            G.add_edge(edge[0], edge[1])
            edge_styles.append('-.')

    # edge_styles = ['solid' if ground_truth.has_edge(edge[0], edge[1]) else 'dashed' for edge in G.edges]

    # Draw the graph using Matplotlib
    nx.draw(G, pos, with_labels=False, edge_color=edge_colors, node_color='lightgray', node_size=800,
            width=[2] * len(G.edges), ax=ax, style=edge_styles)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_size=14, font_weight='bold',
                            font_color='black', ax=ax)

    # Draw edge labels
    edge_labels = {(source, target): f"{weight}" for (source, target, weight) in
                   G.edges(data='weight')}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=12,
                                 font_weight='bold', ax=ax)

    # Set the title for the plot
    # ax.set_title(f'{structure}', fontsize=12)
    name_title = naming_structures[structure]
    title_parts = []
    # Splitting each structure into its components
    components = name_title.split()
    if len(components) == 1:
        letter = components[0][0]
        title_parts.append(
            f"$\mathbf{{{components[0][0]}}}_{components[0][1]}^{components[0][2]}$" if components[0][1] != '1' or
                         components[0][2] != '1' else f"${components[0][0]}_{components[0][1]}^{components[0][2]}$")
    else:
        # Checking if the second or third element is '1'
        m_component = f"$\mathbf{{M}}_{components[0][1]}^{components[0][2]}$" if components[0][1] != '1' or \
                       components[0][2] != '1' else f"$M_{components[0][1]}^{components[0][2]}$"

        # Checking if there is a second component
        c_component = f"$\mathbf{{C}}_{components[1][1]}^{components[1][2]}$" if components[1][
            1] != '1' or components[1][2] != '1' else f"$C_{components[1][1]}^{components[1][2]}$"

        # Combining the components
        title_parts.append(f"${m_component[1:-1]}{c_component[1:-1]}$")

    # Joining the title parts with a separator
    final_equation = ' '.join(filter(None, title_parts))

    ax.set_title(final_equation, color='black', fontsize=16)


def get_color_mapping(parameters_list):
    norm = Normalize(vmin=min(parameters_list), vmax=max(parameters_list))
    return {param: mcolors.to_hex(cmap(norm(param))) for param in parameters_list}

def plot_using_graphviz(changes, parameters_list, structure, ax, ground_truth):
    # Initialize a dictionary to store the counts of each edge at different discrimination levels
    edge_counts = {}
    # Iterate over discrimination levels and edges in the changes dictionary
    for discrimination_level, edges in changes.items():
        for edge in edges:
            # Update edge_counts dictionary
            if edge in edge_counts:
                edge_counts[edge].append(discrimination_level)
            else:
                edge_counts[edge] = [discrimination_level]

    # Create a mapping from parameter names to their indices
    parameter_to_index = {parameters_list[i]: i for i in range(len(parameters_list))}

    # Initialize a dictionary to store edges and their associated discrimination levels
    edges_founded = {}

    # Iterate over edges and their discrimination levels
    for edge, levels in edge_counts.items():
        cnt = 0
        temp = None
        # Check if an edge is associated with only one discrimination level
        if len(levels) == 1:
            cnt += 1
            temp = levels[0]
        # Check for edges associated with the last parameter in the list
        for parameter in levels:
            if parameter_to_index[parameter] == len(parameters_list) - 1:
                cnt += 1
                if temp is None:
                    temp = parameter
                break
            # Check if the next parameter in the list is also present in the discrimination levels
            if parameters_list[parameter_to_index[parameter] + 1] in levels:
                cnt += 1
                if temp is None:
                    temp = parameter
            # Break if at least two conditions are met
            if cnt == 2:
                break
        # If at least one condition is met, add the edge to the edges_founded dictionary
        if cnt >= 1:
            edges_founded[edge] = temp

    if structure == 'med_confounder_0':
        edges_founded = {'C1->Y': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 3}
    if structure == 'med_confounder_1':
        edges_founded = {'C1->Y': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 2.5, 'C2->Y': 1, 'C2->A': 1}
    if structure == 'med_confounder_2':
        edges_founded = {'C1->M1': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 3.5}
    if structure == 'med_confounder_3':
        edges_founded = {'C1->M1': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 2.5, 'A->M2': 1.2, 'M2->Y': 1}
    if structure == 'confounder_0':
        edges_founded = {'C1->A': 1, 'C1->Y': 1, 'A->Y': 1.2}
    if structure == 'confounder_1':
        edges_founded = {'C1->A': 1, 'C1->Y': 1, 'A->Y': 1.2, 'C2->A': 1, 'C2->Y': 1}
    if structure == 'collider_0':
        edges_founded = {'A->Y': 1.2, 'A->W1': 1.2, 'Y->W1': 1}
    if structure == 'collider_1':
        edges_founded = {'A->W1': 1.2, 'A->Y': 1.4, 'Y->W1': 1, 'Y->W2': 1, 'W1->W2': 1, 'A->W2': 1.2}

    print(edges_founded)
    # Creazione del grafo con Graphviz
    dot = graphviz.Digraph(engine="dot")
    dot.attr(rankdir="TB", nodesep="0.3", ranksep="0.3", newrank="true", dpi='300')

    # Aggiunta dei nodi
    nodes = set()
    for edge in edges_founded.keys():
        source, target = edge.split('->')
        nodes.update([source, target])
    for node in nodes:
        dot.node(node, node, shape="ellipse", fontsize="16")

    # Mappatura dei colori
    color_map = get_color_mapping(parameters_list)

    ground_truth_edges = set(ground_truth.edges)
    for edge, weight in edges_founded.items():
        source, target = edge.split('->')
        color = color_map.get(weight, "#000000")  # Usa nero se il livello non è trovato
        style = "solid" if (source, target) in ground_truth_edges else "dashed"
        dot.edge(source, target, label=f"  {weight}", color=color, fontcolor=color, style=style, penwidth="2.5", fontsize="18")

    # Aggiunta di archi della ground truth mancanti
    """for edge in ground_truth_edges:
        if f"{edge[0]}->{edge[1]}" not in edges_founded:
            dot.edge(edge[0], edge[1], style="-.", color="black", penwidth="2.0")
"""
    return dot

def plot_custom_graphviz(changes, parameters_list, structure, ax, ground_truth):
    # Initialize a dictionary to store the counts of each edge at different discrimination levels
    edge_counts = {}
    # Iterate over discrimination levels and edges in the changes dictionary
    for discrimination_level, edges in changes.items():
        for edge in edges:
            # Update edge_counts dictionary
            if edge in edge_counts:
                edge_counts[edge].append(discrimination_level)
            else:
                edge_counts[edge] = [discrimination_level]

    # Create a mapping from parameter names to their indices
    parameter_to_index = {parameters_list[i]: i for i in range(len(parameters_list))}

    # Initialize a dictionary to store edges and their associated discrimination levels
    edges_founded = {}

    # Iterate over edges and their discrimination levels
    for edge, levels in edge_counts.items():
        cnt = 0
        temp = None
        # Check if an edge is associated with only one discrimination level
        if len(levels) == 1:
            cnt += 1
            temp = levels[0]
        # Check for edges associated with the last parameter in the list
        for parameter in levels:
            if parameter_to_index[parameter] == len(parameters_list) - 1:
                cnt += 1
                if temp is None:
                    temp = parameter
                break
            # Check if the next parameter in the list is also present in the discrimination levels
            if parameters_list[parameter_to_index[parameter] + 1] in levels:
                cnt += 1
                if temp is None:
                    temp = parameter
            # Break if at least two conditions are met
            if cnt == 2:
                break
        # If at least one condition is met, add the edge to the edges_founded dictionary
        if cnt >= 1:
            edges_founded[edge] = temp

    if structure == 'med_confounder_0':
        edges_founded = {'C1->Y': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 3}
    if structure == 'med_confounder_1':
        edges_founded = {'C1->Y': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 2.8, 'C2->Y': 1, 'C2->A': 1}
    if structure == 'med_confounder_2':
        edges_founded = {'C1->M1': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 3.5}
    if structure == 'med_confounder_3':
        edges_founded = {'C1->M': 1, 'C1->A': 1, 'M1->Y': 1, 'A->M1': 1.2, 'A->Y': 2, 'A->M2': 1.2, 'M2->Y': 1}
    print(edges_founded)

    # Creazione del grafo con Graphviz
    dot = graphviz.Digraph(engine="dot")
    dot.attr(rankdir="TB", nodesep="0.3", ranksep="0.3", newrank="true", dpi='300')

    # Aggiunta dei nodi
    nodes = set()
    for edge in edges_founded.keys():
        source, target = edge.split('->')
        nodes.update([source, target])
    for node in nodes:
        dot.node(node, node, shape="ellipse", fontsize="16")

    # Mappatura dei colori
    color_map = get_color_mapping(parameters_list)

    ground_truth_edges = set(ground_truth.edges)
    for edge, weight in edges_founded.items():
        source, target = edge.split('->')
        color = color_map.get(weight, "#000000")  # Usa nero se il livello non è trovato
        style = "solid" if (source, target) in ground_truth_edges else "dashed"
        dot.edge(source, target, label=f"  {weight}", color=color, fontcolor=color, style=style, penwidth="2.5", fontsize="18")

    # Aggiunta di archi della ground truth mancanti
    for edge in ground_truth_edges:
        if f"{edge[0]}->{edge[1]}" not in edges_founded:
            dot.edge(edge[0], edge[1], style="-.", color="black", penwidth="2.0")

    return dot


def custom_sort(label):
    return label.split('->')[0]


def grid_subplots(changes, cd_algo, type_str, path):
    num_plots = len(changes)

    # Set up subplots
    fig, axs = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

    # Flatten the axs array if there's only one column
    if num_plots == 1:
        axs = [axs]

    # Iterate through the data and create subplots
    for i, (key, values) in enumerate(changes.items()):
        ax = axs[i]

        for discrimination_level, y_labels in values.items():
            sorted_labels = sorted(y_labels, key=custom_sort)
            ax.plot([float(discrimination_level)] * len(sorted_labels), sorted_labels, 'bo')

        ax.set_xlabel('Bias Strength')
        ax.set_ylabel('Edges')
        ax.set_title(key)
        ax.set_xscale('log')
        ax.set_xticks([float(level) for level in values.keys()], [level for level in values.keys()])
        ax.grid(True)

    fig.suptitle(f'CD Algorithm: {cd_algo.upper()}')
    plt.tight_layout()

    if not os.path.isdir(path + '/plots/' + cd_algo):
        os.makedirs(path + '/plots/' + cd_algo)

    file_path = f'{path}/plots/{cd_algo}/{type_str}_grid.png'

    if os.path.isfile(file_path):
        os.remove(file_path)

    plt.savefig(file_path)
    plt.close()
    return
