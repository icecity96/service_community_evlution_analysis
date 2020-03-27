import networkx as nx


def community_density(graph: nx.Graph, community: list) -> float:
    """
    $density = \frac{E}{n \times (n-1)}$ where E is the number of edges in the community and n is the number of nodes
    in the community
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :return: community density score
    """
    subgraph = graph.subgraph(community)
    density = subgraph.number_of_edges() / (len(community)**2 - len(community))
    return density


def community_clustering(graph: nx.Graph, community: list) -> float:
    """
    community clustering score is the mean of nodes clustering score
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :return: community clustering score
    """
    clustering_coefficient = nx.clustering(graph, community, "weight")
    score = sum(list(clustering_coefficient.values())) / len(community)
    return score


def community_average_closeness_centrality(graph: nx.Graph, community: list, closeness: str = "closeness") -> float:
    """
    community closeness centrality score is the mean of nodes closeness centrality score.
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :return: community closeness centrality score
    """
    if closeness not in graph.nodes[community[0]]:
        closeness_centrality = dict(nx.closeness_centrality(graph))
        for key, value in closeness_centrality.items():
            graph.nodes[key][closeness] = value
    return sum([graph.nodes[node][closeness] for node in community]) / len(community)


def community_degree(graph: nx.Graph, community: list) -> float:
    """
    community degree score is the mean of nodes degree score
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :return: community degree centrality score
    """
    degrees = [graph.degree(node, "weight") for node in community]
    score = sum(degrees) / len(community)
    return score


def community_eigenvector_centrality(graph: nx.Graph, community: list, eigenvector: str = "eigenvector") -> float:
    """
    community eigenvector score is the mean of nodes eigenvector score
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :param eigenvector: where to store the values
    :return: community eigenvector centrality score
    """
    if eigenvector not in graph.nodes[community[0]]:
        centrality = nx.eigenvector_centrality(graph, weight="weight")
        for key, value in centrality.items():
            graph.nodes[key][eigenvector] = value
    return sum([graph.nodes[node][eigenvector] for node in community]) / len(community)


def community_leadership(graph: nx.Graph, community: list) -> float:
    """
    community leadership score reflects the centralization of the community.
    :param graph: the graph object to be used
    :param community: a list of nodes in the community. the size of the community must be greater than or equal to 3
    :return: community leadership score
    """
    node_degrees = [dv[1] for dv in graph.degree(community, "weight")]
    max_degree = max(node_degrees)
    denomin = (len(community) - 2) * (len(community) - 1)
    return sum([max_degree - v for v in node_degrees]) / denomin


def community_tpratio(graph: nx.Graph, community: list) -> dict:
    """
    Calculate the ratio of each type of nodes in the community
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :return: a dictionary with keys representing node type, values representing the ratio
    """
    tpratio = {}
    for node in community:
        node_type = graph.nodes[node]['type']
        tpratio[node_type] = tpratio.get(node, 0) + 1
    for key, value in tpratio.items():
        tpratio[key] = value/len(community)
    return tpratio


def community_activity(graph: nx.Graph, community: list) -> tuple:
    """
    Calculate the activity score of the community
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :return: a tuple(max, mean, sum)
    """
    subgraph = graph.subgraph(community)
    activity_scores = [d for u, v, d in subgraph.edges(data="weight")]
    return max(activity_scores), sum(activity_scores) / len(community), sum(activity_scores)


def community_keynodes(graph: nx.Graph, community: list, social_position: list, alpha: int = 10) -> list:
    """
    Key nodes detection Algorithm
    :param graph: the graph object to be used
    :param community: a list of nodes in the community
    :param social_position: social position score of each node in the community
    :param alpha: therehold score
    :return: a list of key nodes
    """
    if len(set(social_position)) == 1:  # SP(V1) == SP(V2) == ... == SP(Vn)
        return community
    key, SP = {}, {}
    for node, sp in zip(community, social_position):
        key[node], SP[node] = 0, sp

    subgraph = graph.subgraph(community)
    for u, v in subgraph.edges():
        if SP[u] < SP[v]:
            key[u] -= abs(SP[u]-SP[v])
            key[v] += abs(SP[v]-SP[u])
        else:
            key[u] += abs(SP[u]-SP[v])
            key[v] -= abs(SP[v]-SP[u])
    max_score = max(list(key.values()))
    keynodes =[node for node, score in key.items() if score > 0 and max_score / score < alpha]
    return keynodes


def community_cohesion(graph: nx.Graph, community: list) -> float:
    """
    cohesion measure characterising strength of connections inside group in relation to connections outside group
    $cohesion = \frac{\frac{\sum_{i \in C}\sum_{j \in C}w(i,j)}{n(n-1)}}{\frac{\sum_{i \in C}\sum_{j not \in C}w(i,j)}{N(N-n)}}$
    :param graph: The graph to be used
    :param community: a list of nodes in the community
    :return: cohesion score
    """
    N, n = graph.number_of_nodes(), len(community)
    inter, outer = 0, 0
    for u, v, w in graph.edges(data="weight"):
        flags = [u in community, v in community]
        if all(flags):
            inter += w
        elif any(flags):
            outer += w
    cohesion = 0
    if outer == 0:
        cohesion = 10000    # 10000 is Large enough
    else:
        cohesion = (inter / (n * (n - 1))) / (outer / N * (N - n))
    return cohesion
