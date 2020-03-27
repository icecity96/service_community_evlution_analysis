import networkx as nx
from cdlib import algorithms
from community_features import *
import pickle


def static_community_detection(snapshots, pkl=None) -> list:
    """
    Detect community structure in each snapshot
    :param snapshots: generated snapshots
    :param pkl: pickle file path to store the community detection results
    :return list of community structure in each snapshot.
            And by default, detection results will be written to data/communities.pkl
    """
    if pkl is not None:
        print(f"loading communities from {pkl}")
        return pickle.load(open(pkl, 'rb'))
    snapshots = snapshots["snapshots"]
    communities = []
    for snapshot in snapshots:
        communities.append(algorithms.louvain(snapshot, 'weight'))
    # only stable communities will be included
    for index, community_struct in enumerate(communities):
        community_struct.communities = [c for c in community_struct.communities if len(c) >= 3]
        communities[index] = community_struct
    pickle.dump(communities, open("data/communities.pkl", "wb"))
    return communities


def social_position_score(snapshots, pkl=None) -> list:
    """
    Calculate the social position score of each node in each snapshot (In this paper we use PageRank score)
    :param snapshots: generated snapshots
    :param pkl: a pickle file storing the social position score
    :return: list of social position scores in each snapshot.[{node: score, ...},...]
             And by default, the results will be stored to data/social_position.pkl
    """
    if pkl is not None:
        print(f"loading social position score from {pkl}")
        return pickle.load(open(pkl, 'rb'))
    snapshots, social_positions = snapshots["snapshots"], []
    for snapshot in snapshots:
        page_rank_score = nx.pagerank(snapshot, alpha=0.85, weight="weight")
        social_positions.append(dict(page_rank_score))
    pickle.dump(social_positions, open("data/social_positions.pkl", "wb"))
    return social_positions


def _inclusion(C1: list, C2: list, SP1: dict) -> float:
    """
    *inclusion* allows to evaluate the inclusion of one community in another.
    $I(C1, C2) = \frac{|C_1 \cap C_2|}{|G_1|}$ \frac{\sum_{x \in (G_1 \cap C_2)}SP_{G_1}(x)}{\sum_{x \in G_1}SP_{G_1}(x)}
    :param C1: community 1
    :param C2: community 2
    :param SP1: social_position of nodes in the C1
    :return: inclusion socre
    """
    quantity = len(set(C1) & set(C2)) / len(C1)
    quality = sum([SP1[node] for node in list(set(C1) & set(C2))])
    quality /= sum([SP1[node] for node in C1])
    return quantity * quality


def _event_identifier(C1, C2, SP1, SP2, alpha=0.5, beta=0.6):
    """
    based on both inclusion I(C1,C2) and I(C2,C1)
    :param C1:
    :param C2:
    :param SP1:
    :param SP2:
    :return:
    """
    I1, I2 = _inclusion(C1, C2, SP1), _inclusion(C2, C1, SP2)
    # Continuing
    if I1 >= alpha and I2 >= beta and len(C1) == len(C2):
        return "continuing"

    # shrinking
    if (I1 >= alpha and I2 >= beta and len(C1) > len(C2)) or (I1 < alpha and I2 >= beta and len(C1) >= len(C2)):
        return "shrinking"

    # growing
    if (I1 > alpha and I2 > beta and len(C1) < len(C2)) or (I1 >= alpha and I2 < beta and len(C1) <= len(C2)):
        return "growing"

    # splitting
    if I1 < alpha and I2 >= beta and len(C1) >= len(C2):
        return "splitting"

    # Merging
    if I1 >= alpha and I2 <= beta and len(C1) <= len(C2):
        return "merging"

    return None


def GED(communities1: list, communities2: list, SP1: dict, SP2: dict, alpha: float, beta: float):
    """
    Group Evolution Discovery method
    :param beta:
    :param alpha:
    :param communities1:
    :param communities2:
    :param SP1:
    :param SP2:
    :return:
    """
    pre_window_event, next_window_event = {}, {}
    possible_events = []
    for i, community1 in enumerate(communities1):
        if "A-{:d}".format(i) not in next_window_event:
            next_window_event["A-{:d}".format(i)] = []
        for j, community2 in enumerate(communities2):
            if "B-{:d}".format(j) not in pre_window_event:
                pre_window_event["B-{:d}".format(j)] = []
            event = _event_identifier(community1, community2, SP1, SP2, alpha, beta)
            if event is None:
                continue
            next_window_event["A-{:d}".format(i)].append(event)
            pre_window_event["B-{:d}".format(j)].append(event)
            possible_events.append(("A-{:d}".format(i), "B-{:d}".format(j), event))
    events = []
    for key, value in next_window_event.items():
        if len(value) == 0:
            events.append((key, "dissolving"))
        if len(value) == 1 and value[0] == "shrinking":
            events.append((key, "shrinking"))
        if len(value) == 1 and value[0] == "continuing":
            events.append((key, "continuing"))
        if len(value) > 1:
            events.append((key, "splitting"))

    for key, value in pre_window_event.items():
        if len(value) == 0:
            events.append((key, "forming"))
        if len(value) == 1 and value[0] == "growing":
            events.append((key, "growing"))
        if len(value) > 1:
            events.append((key, "merging"))
    return possible_events, events


def meta_community_network_generation(communities, social_positions, alpha=None, beta=None, pkl=None) -> nx.DiGraph:
    """
    construct a meta community network.
    :param beta:
    :param alpha:
    :param communities:
    :param social_positions:
    :param pkl:
    :return:
    """
    if pkl is not None:
        print(f"loading meta community network from {pkl}")
        return pickle.load(open(pkl, 'rb'))
    meta_community_network = nx.DiGraph()

    for index, community_struct in enumerate(communities):
        for index_j, community in enumerate(community_struct.communities):
            meta_community_network.add_node(f"T{index}C{index_j}", pre="None", nex="None")

    for index in range(len(communities) - 1):
        C1, C2 = communities[index].communities, communities[index + 1].communities
        SP1, SP2 = social_positions[index], social_positions[index + 1]
        possible_events, events = GED(C1, C2, SP1, SP2, alpha, beta)
        for possible_event in possible_events:
            source, target = possible_event[0], possible_event[1]
            source = "T{:d}C".format(index) + source[2:]
            target = "T{:d}C".format(index + 1) + target[2:]
            meta_community_network.add_edge(source, target)
        for event in events:
            node, event_type = event[0], event[1]
            if node[0] == 'A':
                node = "T{:d}C".format(index) + node[2:]
                meta_community_network.nodes[node]["nex"] = event_type
            else:
                node = "T{:d}C".format(index + 1) + node[2:]
                meta_community_network.nodes[node]["pre"] = event_type

    pickle.dump(meta_community_network, open("data/meta_community_network.pkl", "wb"))
    return meta_community_network


FEATURE_NAMES = [
    "size", "density", "clustering", "avg_closeness_centrality", "degree",
    # "eigenvectors_centrality",
    "leadership", "cohesion", "#Keynodes", "max_activity", "mean_activity", "sum_activity", "%Stakeholder",
    "%Service", "Kdegree", "Kavg_closeness_centrality", "Keigenvectors_centrality"
]


def feature_extraction(snapshots, communities, social_positions, pkl=None):
    """
    extract features for each community
    :param pkl:
    :param snapshots:
    :param communities:
    :param social_positions:
    :return:
    """
    if pkl is not None:
        print(f"loading features from {pkl}")
        return pickle.load(open(pkl, 'rb'))
    features, snapshots = [], snapshots["snapshots"]
    for snapshot, community_struct, social_position in zip(snapshots, communities, social_positions):
        communities_features = []
        for community in community_struct.communities:
            tpratio = community_tpratio(snapshot, community)
            keynodes = community_keynodes(snapshot, community, [social_position[node] for node in community])
            activity = community_activity(snapshot, community)
            community_features = [
                len(community),  # community size
                community_density(snapshot, community),  # community density
                community_clustering(snapshot, community),  # community clustering
                community_average_closeness_centrality(snapshot, community),  # average closeness centrality
                community_degree(snapshot, community),  # community degree
                # community_eigenvector_centrality(snapshot, community),  # eigenvector centrality
                community_leadership(snapshot, community),  # community leadership
                community_cohesion(snapshot, community),  # community cohesion
                len(keynodes),  # number of keynodes
                activity[0],  # max activity
                activity[1],  # mean activity
                activity[2],  # sum activity
                tpratio.get("Stakeholder", 0),  # number of stakeholders in community
                tpratio.get("Service", 0),  # number of services in community
                community_degree(snapshot, keynodes),  # key nodes degree
                community_average_closeness_centrality(snapshot, keynodes),  # key nodes average closeness
                # community_eigenvector_centrality(snapshot, keynodes),  # key nodes eigenvectors centrality
            ]
            communities_features.append(community_features.copy())
        features.append(communities_features.copy())
    pickle.dump(features, open("data/features.pkl", "wb"))
    return features
