import json
import networkx as nx
from  datetime import datetime as dt
from datetime import timedelta
import pickle


INITIAL_WEIGHT = {"exit": -1, "conflict": -1, "join": 4, "acquisition": 4, "BelongTo": 2, "Structural": 2}
STABLE_EDGE = ["exit", "conflict", "join", "acquisition", "BelongTo", "Structural"]


def linear_decay(edge_type: str, duration: int, initial_weight: dict = None,
                 stable_edge: list = None, decay_rate: int = 30, default_weight: float = 2.5, max_decay: int = 12):
    """
    A linear decay for edge weights in the dynamic network
    :param edge_type: a string representing edge type
    :param duration: how long this edge created (must be days for our project)
    :param initial_weight: initial weight for specified edge types
    :param stable_edge: if an edge type in stable_edge, then this type will not decay.
    :param decay_rate: how often to decay (days)
    :param default_weight: default weight for edge types not in initial_weight
    :param max_decay: max decay numbers
    :return: a float value. If return a negative value, it means that the edge should be deleted?
    """
    if stable_edge is None:
        stable_edge = STABLE_EDGE
    if initial_weight is None:
        initial_weight = INITIAL_WEIGHT
    if edge_type in stable_edge:
        return initial_weight.get(edge_type, default_weight)

    weight = initial_weight.get(edge_type, default_weight)
    aging_cofficient = max(duration / decay_rate, 1)
    if aging_cofficient > max_decay:
        return -1
    return weight / aging_cofficient


def _generate_snapshot(end_time, nodes, edges, ignore_event=True):
    """
    generate a snapshot at end time
    :param end_time: a timestamp
    :param nodes: node list
    :param edges: edge list
    :param ignore_event: True if don't care about events
    :return: a networkx graph object to present the snapshot
    """
    snapshot = nx.Graph()
    for node in nodes:
        if ignore_event and node["type"] == "Event":
            continue
        snapshot.add_node(node["id"], type=node["type"])

    edges_retain = []
    for edge in edges:
        if "timestamp" not in edge:
            edge["timestamp"] = "1990-01-01"
        if dt.strptime(edge["timestamp"], "%Y-%m-%d") <= end_time:
            edges_retain.append(edge)
    edges_retain = sorted(edges_retain, key=lambda x: dt.strptime(x['timestamp'], '%Y-%m-%d'))

    for edge in edges_retain:
        if not snapshot.has_node(edge['source']) or not snapshot.has_node(edge['target']):
            continue
        duration = (end_time - dt.strptime(edge['timestamp'], '%Y-%m-%d')).days
        r = edge['r'] if edge['type'] != 'Structural' else "Structural"
        weight = linear_decay(r, duration)
        if weight < 0:
            if snapshot.has_edge(edge['source'], edge['target']):
                snapshot.remove_edge(edge['source'], edge['target'])
        else:
            if not snapshot.has_edge(edge['source'], edge['target']):
                snapshot.add_edge(edge['source'], edge['target'], weight=0)
            snapshot[edge['source']][edge['target']]['weight'] += weight

    snapshot.remove_nodes_from(list(nx.isolates(snapshot)))
    return snapshot


def generate_snapshots(end_time: dt, window_size: int, edges: list, nodes: list, pkl=None):
    """
    generate a set of snapshots at different timestamp
    :param end_time: a timestamp
    :param window_size: days between snapshots
    :param edges: a list of edges
    :param nodes: a list of nodes
    :param pkl: if provided, then will load from it. otherwise will generate step by step
    :return: a dict {"snapshot": [snapshot...], "timestamp": [end_time...]}. And will generate a
            pkl file in "data/snapshots.pkl"
    """
    if pkl is not None:
        print(f"loading snapshots from {pkl}")
        return pickle.load(open(pkl, "rb"))
    snapshots, timestamps = [], []
    while end_time < dt.strptime("2019-12-30", "%Y-%m-%d"):
        snapshots.append(_generate_snapshot(end_time, nodes, edges))
        timestamps.append(end_time.strftime("%Y-%m-%d"))
        end_time += timedelta(days=window_size)
    pickle.dump({"snapshots": snapshots, "timestamps": timestamps}, open("data/snapshots.pkl", "wb"))
    return {"snapshots": snapshots, "timestamps": timestamps}
