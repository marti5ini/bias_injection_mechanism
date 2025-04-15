import random

import networkx as nx
import numpy as np
import pandas as pd


class CausalDataGenerator:

    def __init__(self, dag, sensitive_attributes=['A'], acceptance_rate=0.5, y_binary=True,
                 discrimination_rank_list=None, noise_scale=0.1, parameter1=0.1, parameter2=0,
                 outcome_name='Y', coefficients_range=[0, 1], sem_type='gauss', model_type='linear',
                 weigths=True
                 ):
        """
        Constructs an instance of the Generator class for generating
        synthetic biased data based on a causal graph.

        :param dag: networkx.DiGraph
            A directed graph representing the causal structure
        :param sensitive_attributes: list, (default=['A'])
            List for the sensitive attributes names
        :param noise_scale: float, (default: 0.1)
            Maximum amount of random noise added to the generated data
        :param parameter1: int, (default: 3)
            Parameter 1 used in creating bias
        :param parameter2: int, (default: 4)
            Parameter 2 used in creating bias
        :param discrimination_rank_list: list, default=None
            The list should contain the ordered sequence of classes of the sensitive attribute.
            The least discriminated against class first.
        """
        self.structural_equations = dict()
        self.graph = dag
        assert nx.is_directed_acyclic_graph(self.graph), "The causal graph must be a DAG."
        self.sensitive_attributes = sensitive_attributes
        assert len(sensitive_attributes) >= 1, "At least one sensitive attribute is required."
        assert 'Y' in dag.nodes, ("The default outcome node name is not present in the DAG. "
                                  "Change the parameter.")
        self.outcome_node = outcome_name
        self.discriminated_attributes = {node: list(self.graph.successors(node)) for node in sensitive_attributes}
        self.noise_scale = noise_scale
        self.param1 = parameter1
        self.param2 = parameter2 #1 - np.log(parameter2 + 1)
        self.higher_param = max(self.param1, self.param2)
        self.lower_param = min(self.param1, self.param2)
        # self.seed = hash(int(time.time()))
        self.discrimination_rank_list = discrimination_rank_list
        # mapping class names to class numbers in order to match them with
        # the generated values from the distribution
        self.rank_wise_class_numbers = [i for i, rank in
                                        enumerate(discrimination_rank_list)] if discrimination_rank_list else None
        self.acceptance_rate = (1 - acceptance_rate) * 100
        self.y_binary = y_binary
        self.sem_type = sem_type
        self.model_type = model_type
        self.low, self.high = coefficients_range
        self.parents = {node: list(self.graph.predecessors(node)) for node in self.graph}
        self.confounders = self.get_confounders(self.parents)
        self.weights = weigths


    def generate_random_noise(self, n_samples):
        if self.sem_type == 'gauss':
            noise = np.random.normal(scale=self.noise_scale, size=n_samples)
        elif self.sem_type == 'uniform':
            noise = np.random.uniform(high=self.noise_scale, size=n_samples)
        elif self.sem_type == 'exp':
            noise = np.random.exponential(scale=self.noise_scale, size=n_samples)
        else:
            raise ValueError('sem_type not implemented yet.')

        return noise

    def _get_weights(self, parents):
        if self.weights:
            weights = np.random.uniform(low=self.low, high=self.high, size=len(parents)).tolist()
        else:
            weights = [1] * len(parents)
        return weights

    def learn_structural_equation(self, n_samples, parents_name):
        parents = np.array([self.structural_equations[node] for node in parents_name])
        if self.model_type == 'linear':
            weights = self._get_weights(parents)
            result = weights @ parents
            result += self.generate_random_noise(n_samples)
        elif self.model_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            result = gp.sample_y(parents, random_state=None).flatten()
            result += np.random.normal(scale=self.noise_scale, size=n_samples)
        else:
            raise ValueError('unknown model_type.')
        return result

    def sensitive_child(self, n_samples, parents):
        """
        Generates a distribution for a node with incoming edges from the sensitive attribute.

        :param n_samples: int
            Number of samples to generate
        :param parents: Tuple
            Incoming edges from sensitive attributes
        :return: numpy.ndarray
            Generated samples for the node
        """
        non_sensitive_parents = np.array(
            [self.structural_equations[node] for node in parents if node not in self.sensitive_attributes])
        sensitive_parents = np.array(
            [self.structural_equations[node] for node in parents if node in self.sensitive_attributes])

        if len(non_sensitive_parents):
            weights = self._get_weights(non_sensitive_parents)
            result = weights @ non_sensitive_parents
        else:
            result = np.array([0.0] * n_samples)

        for parent in sensitive_parents:
            weight = self._get_weights([1])[0]
            for i in range(len(parent)):
                result[i] += weight * (self._generate_unprivileged_sample() if parent[i] == 0
                                       else (self._generate_privileged_sample()))
        result += self.generate_random_noise(n_samples)
        return result

    def confounder_child(self, n_samples, parents, confounder_parents):
        equations = np.array([self.structural_equations[node] for node in parents])

        non_confounder_parents = equations[
            ~np.isin(parents, confounder_parents) & ~np.isin(parents, self.sensitive_attributes)]
        sensitive_parents = equations[np.isin(parents, self.sensitive_attributes)]
        confounders = equations[np.isin(parents, confounder_parents) & ~np.isin(parents, self.sensitive_attributes)]

        if len(non_confounder_parents):
            weights = self._get_weights(non_confounder_parents)
            result = weights @ non_confounder_parents
        else:
            result = np.array([0.0] * n_samples)

        if len(sensitive_parents):
            weight = self._get_weights([1])[0]
            for parent in sensitive_parents:
                for i in range(len(parent)):
                    result[i] += weight * (self._generate_unprivileged_sample() if parent[i] == 0
                                           else (self._generate_privileged_sample()))

        for parent in confounders:
            weight = self._get_weights([1])[0]
            for i in range(len(parent)):
                x = np.random.uniform(low=0, high=1)
                if x < parent[i]:
                    result[i] += weight * (self.higher_param * random.uniform(parent[i], 1) +
                                           self.lower_param * random.uniform(0, parent[i]))
                else:
                    result[i] += weight * (self.lower_param * random.uniform(parent[i], 1) +
                                           self.higher_param * random.uniform(0, parent[i]))

        result += self.generate_random_noise(n_samples)
        return result

    def _generate_unprivileged_sample(self):
        # return (self.higher_param * random.uniform(0, 0.5) +
        #        self.lower_param * random.uniform(0.5, 1))
        return (self.higher_param * np.random.normal(0.25, 0.1) +
                self.lower_param * np.random.normal(0.75, 0.1))

    def _generate_privileged_sample(self):
        # return (self.lower_param * random.uniform(0, 0.5) +
        #         self.higher_param * random.uniform(0.5, 1))
        return (self.lower_param * np.random.normal(0.25, 0.1) +
                self.higher_param * np.random.normal(0.75, 0.1))

    def is_discriminated(self, node):
        return any(node in discriminated_attrs for discriminated_attrs in self.discriminated_attributes.values())

    def get_confounders(self, parents):
        confounders = {sensitive_att: [] for sensitive_att in self.sensitive_attributes}
        sensitive_parents = {sensitive_att: list(self.graph.predecessors(sensitive_att))
                             for sensitive_att in self.sensitive_attributes}
        for sensitive_name, sa_parents in sensitive_parents.items():
            if len(sa_parents):
                for node in parents:
                    if node not in self.sensitive_attributes:
                        is_confounder = sa_parents and parents[node]
                        if len(is_confounder):
                            confounders[sensitive_name] += is_confounder
        return confounders

    def generate_data(self, n_samples=1000):
        """
        Fits the Generator model and generates synthetic data.
        :param n_samples: int
            Number of samples to generate
        :return: pd.DataFrame
            Generated synthetic data
        """
        for node_name in nx.topological_sort(self.graph):

            # The node is a root
            if not self.parents[node_name]:
                # The node is a sensitive attribute
                if node_name in self.sensitive_attributes:
                    self.structural_equations[node_name] = np.random.binomial(1, 0.5, size=n_samples)
                else:
                    self.structural_equations[node_name] = np.random.beta(3, 3, size=n_samples)
            else:
                confounder_parents = [parent
                                      for sensitive_attr in self.sensitive_attributes
                                      for parent in self.parents[node_name]
                                      if parent in self.confounders[sensitive_attr]]
                # The node is a sensitive attribute
                if node_name in self.sensitive_attributes:
                    result = normalizer(self.learn_structural_equation(n_samples, self.parents[node_name]))
                    self.structural_equations[node_name] = np.random.binomial(1, result)
                # The node is not a sensitive attribute and is a child of a confounding parent node
                elif node_name not in self.sensitive_attributes and len(confounder_parents):
                    self.structural_equations[node_name] = normalizer(self.confounder_child(n_samples,
                                                                                            self.parents[node_name],
                                                                                            confounder_parents))
                # The node is a child of the sensitive attribute (and of other parent nodes if they exist)
                elif self.is_discriminated(node_name):
                    self.structural_equations[node_name] = normalizer(self.sensitive_child(n_samples,
                                                                                           self.parents[node_name]))
                # The node is a child of a non-sensitive attribute
                else:
                    self.structural_equations[node_name] = normalizer(self.learn_structural_equation(n_samples,
                                                                                                     self.parents[
                                                                                                         node_name]))

        df = pd.DataFrame(self.structural_equations)

        if self.y_binary:
            df[self.outcome_node] = np.where(df[self.outcome_node] >= np.percentile(df[self.outcome_node],
                                                                                    self.acceptance_rate), 1, 0)

        return df


def normalizer(result):
    minimum, maximum = np.min(result), np.max(result)
    return (result - minimum) / (maximum - minimum)


if __name__ == '__main__':
    graph = nx.DiGraph()
    graph.add_nodes_from(['A', 'Y', 'M'])
    graph.add_edges_from([('Y' 'A'), ('M', 'Y')])
    model = CausalDataGenerator(graph, acceptance_rate=0.5, parameter1=1, parameter2=1, weigths=False)
    data = model.generate_data(1000)
    print(data[0:20])
    print(data.describe())
