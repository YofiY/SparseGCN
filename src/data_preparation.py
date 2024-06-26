import numpy as np
import scipy 
import networkx as nx
import rustworkx as rwx
import random
import torch
import pygsp

def add_weights_to_edges(G, method: str):
    """
    Add weights to the edges of a graph based on a specified similarity measure transformed into a distance measure.

    This function takes a NetworkX graph and a specified method to calculate edge weights.
    It then copies the graph, removes self-loops, calculates the edge weights using the specified method,
    and sets these weights as edge attributes in the copied graph.

    Parameters:
    ----------
    G : networkx.Graph
        The input graph. It should be a NetworkX graph object.
        
    method : str
        The method to calculate edge weights. Options are 'jaccard' for Jaccard coefficient or
        'adamic_adar' for Adamic-Adar index.

    Returns:
    -------
    networkx.Graph
        The graph with updated edge weights.

    Methods:
    -------
    jaccard:
        The Jaccard coefficient is a similarity measure calculated for each edge. The weight for each edge (u, v) is transformed into a distance measure using the formula:
        1 / (J + 1 / len(set(G[u]) | set(G[v]))) + 1, where J is the Jaccard coefficient.
        
    adamic_adar:
        The Adamic-Adar index is a similarity measure calculated for each edge. The weight for each edge (u, v) is transformed into a distance measure using the formula:
        1 / (A + 1 / len(set(G[u]) | set(G[v]))) + 1, where A is the Adamic-Adar index.

    Notes:
    -----
    - Similarity measures like the Jaccard coefficient and Adamic-Adar index indicate how similar two nodes are. Higher values mean more similarity.
    - To transform these similarity measures into distance measures (where higher values mean more distance), we take their inverses.
    - Self-loops are removed from the graph before calculating edge weights.
    - The function returns a new graph with updated edge weights, leaving the original graph unchanged.

    Examples:
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> G_weighted = add_weights_to_edges(G, method='jaccard')
    """
    
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    
    if method == 'jaccard':
        jaccard_coefs = nx.jaccard_coefficient(G, G.edges)
        jaccard_coefs_dict = {(u, v): (1/(c+1/len(set(G[u]) | set(G[v])))) +1 for u,v,c in jaccard_coefs}
        nx.set_edge_attributes(G, jaccard_coefs_dict, name='weight')  
        
    elif method == 'adamic_adar':
        adamic_adar = nx.adamic_adar_index(G, G.edges)
        adamic_adar_dict = {(u, v): (1/(c+1/len(set(G[u]) | set(G[v])))) +1 for u,v,c in adamic_adar}
        nx.set_edge_attributes(G, adamic_adar_dict, name='weight')
        
    return G


def mb_sparsify(adjacency_matrix):
    """
    Apply metric backbone sparsification to a graph represented by its adjacency matrix.

    This function performs metric backbone sparsification on the input graph. It converts the 
    adjacency matrix to a graph, computes all pairs shortest path (APSP) lengths using Dijkstra's 
    algorithm, and removes edges that are not part of the shortest paths between nodes. The result 
    is a sparser graph that retains the essential structure.

    Parameters:
    ----------
    adjacency_matrix : numpy.ndarray or scipy.sparse.spmatrix
        The input adjacency matrix representing the graph.

    Returns:
    -------
    numpy.ndarray
        The adjacency matrix of the sparsified graph.

    Notes:
    -----
    - Metric backbone sparsification retains only the edges that are part of the shortest paths between nodes, 
      thus simplifying the graph while preserving its core structure.
    - This implementation uses retworkx, a Rust-based graph library, for efficient graph operations.

    Examples:
    --------
    >>> import numpy as np
    >>> adjacency_matrix = np.array([
    ...     [0, 1, 2],
    ...     [1, 0, 1],
    ...     [2, 1, 0]
    ... ])
    >>> sparsified_matrix = mb_sparsify(adjacency_matrix)
    """
    
    g = rwx.PyGraph.from_adjacency_matrix(adjacency_matrix)
    apsp_dict = rwx.all_pairs_dijkstra_path_lengths(g, lambda x: float(x))
    
    for (from_node, to_node) in g.edge_list():
        if (apsp_dict[from_node][to_node] < g.get_edge_data(from_node, to_node)):
            g.remove_edge(from_node, to_node)
    return rwx.adjacency_matrix(g)

def thresh_sparsify(adjacency_matrix, approx_num_edges_wanted):
    a = adjacency_matrix.copy() 
    temp = a.flatten()
    temp = temp[temp != 0]
    threshold = np.percentile(temp, 100 * approx_num_edges_wanted / (len(temp)/2))
    a[a >  threshold] = 0
    return a

def spectral_sparsify(adjacency_matrix:np.ndarray) -> np.ndarray:
    """
    Takes a numpy array, adjacency matrix as an input and returns the adjacency matrix
    of the spectral sparsified version of the corresponding graph using the Spielman and Srivastava sparsification algorithm.

    Args:
        adjacency_matrix (numpy.ndarray): The input adjacency matrix of the graph.

    Returns:
        numpy.ndarray: The adjacency matrix of the spectral sparsified graph.
    """
    g = pygsp.graphs.Graph(adjacency_matrix)
    sparse_g = pygsp.reduction.graph_sparsify(g, epsilon=0.6)
    return sparse_g.W

def random_sparsify(adjacency_matrix, desired_edges):
    """
    Sparsifies a graph given as an adjacency matrix to have approximately the desired number of edges.
    
    Parameters:
    adj_matrix (numpy.ndarray): The adjacency matrix of the graph.
    desired_edges (int): The desired number of edges in the sparsified graph.
    
    Returns:
    numpy.ndarray: The sparsified adjacency matrix.
    """
    n = adjacency_matrix.shape[0]
    
    # Get the list of all edges in the graph
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adjacency_matrix[i, j] > 0]
    
    # If the number of edges in the original graph is less than or equal to the desired number of edges, return the original graph
    if len(edges) <= desired_edges:
        return adjacency_matrix

    # Randomly shuffle the edges
    random.shuffle(edges)
    
    # Select the desired number of edges
    selected_edges = edges[:desired_edges]
    
    # Create a new adjacency matrix with the selected edges
    new_adj_matrix = np.zeros_like(adjacency_matrix)
    for i, j in selected_edges:
        new_adj_matrix[i, j] = adjacency_matrix[i, j]
        new_adj_matrix[j, i] = adjacency_matrix[i, j]
    
    return new_adj_matrix
    
    

 