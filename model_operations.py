import pickle
from typing import List, Tuple

import networkx as nx
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def is_path_valid(path: List[str], graph: nx.DiGraph, SEQUENCE_LENGTH: int = 4) -> bool:
    """
    is this path valid to be an sample
    :param path: a list of nodes
    :param graph: graph object to be used
    :param SEQUENCE_LENGTH: sequence length of path
    :return: boolean
    """
    if len(path) == SEQUENCE_LENGTH:
        return True
    if len(path) == SEQUENCE_LENGTH - 1 and graph.nodes[path[-1]]["nex"] == "dissolving":
        return True
    return False


def extract_ids(community_node: str) -> Tuple[int, int]:
    """
    Get snapshot id and community id from community node
    :param community_node: a string T{sid}C{cid}
    :return: sid, cid
    """
    community_node = community_node.replace("C", "T")
    ids = list(filter(None, community_node.split('T')))
    assert len(ids) == 2
    sid, cid = int(ids[0]), int(ids[1])
    return sid, cid


def _generate_sample_vector(path: list, features, meta_community_network, event_mapping, relative=False):
    sample_X, sample_y = [], [0]
    for index, node in enumerate(path):
        sid, cid = extract_ids(node)
        if index <= 2:
            sample_X.extend(features[sid][cid])
        if index == 2:
            sample_y.append(event_mapping[meta_community_network.nodes[node]['nex']])
        if index == 3:
            sample_y.append(event_mapping[meta_community_network.nodes[node]['pre']])
    if relative:
        length = len(sample_X)
        relative_sample_X = []
        for index, value in enumerate(sample_X):
            if index < length / 3:
                relative_sample_X.append(value)
            else:
                x1, x2 = sample_X[index - int(length / 3)], value
                if x1 != 0:
                    relative_sample_X.append((x2 - x1)/x1 * 100)
                else:
                    relative_sample_X.append(x2 * 100)
        sample_X = relative_sample_X
    sample_y = max(sample_y)
    return sample_X, sample_y


def generate_samples(meta_community_network, features, evolution_type_as_feature=False, pkl=None, relative=False):
    """
    Generate Training and Testing samples from meta-community network
    :param relative:
    :param pkl:
    :param meta_community_network:
    :param features:
    :param evolution_type_as_feature:
    :return:
    """
    if pkl is not None:
        print(f"loading samples from {pkl}")
        return pickle.load(open(pkl, 'rb'))
    paths = dict(nx.all_pairs_shortest_path(meta_community_network, cutoff=4))
    available_paths = [path for single_source_paths in paths.values() for path in single_source_paths.values()
                       if is_path_valid(path, meta_community_network)]
    event_mapping = {"continuing": 1, "growing": 2, "shrinking": 3, "splitting": 4, "merging": 5, "dissolving": 6,
                     "None": 0, "forming": 7}
    X, Y = [], []
    for path in available_paths:
        sample_X, sample_y = _generate_sample_vector(path, features, meta_community_network, event_mapping, relative)
        X.append(sample_X)
        Y.append(sample_y)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
    samples = {"train_X": train_X, "train_Y": train_Y, "test_X": test_X, "test_Y": test_Y}
    pickle.dump(samples, open("data/samples.pkl", "wb"))
    return samples


def train_prediction_model(train_X, train_Y, model_name="random_forest", pkl=None):
    """
    using specified training model
    :param pkl:
    :param train_X:
    :param train_Y:
    :param model_name:
    :return:
    """
    if pkl is not None:
        print(f"loading model from {pkl}")
        return pickle.load(open(pkl, 'rb'))
    model = None
    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=50)
    model.fit(train_X, train_Y)
    explainer = shap.TreeExplainer(model)
    pickle.dump(explainer, open("data/explainer.pkl", "wb"))
    return explainer
