import numpy as np
import networkx as nx
from scipy import spatial


def get_route(start_point: tuple, end_point: tuple, \
              kd_tree: spatial.KDTree, G: nx.Graph, \
              nodes_2_edge_id: dict, point_2_edge: dict, \
              edge_idx_2_nodes: dict, edge_idx_2_points: dict) -> tuple:
    distance_start, index_start = kd_tree.query(start_point)
    distance_end, index_end = kd_tree.query(end_point)

    edge_start, edge_end = point_2_edge[tuple(kd_tree.data[index_start])], \
                           point_2_edge[tuple(kd_tree.data[index_end])]

    edge_nodes_start, edge_nodes_end = edge_idx_2_nodes[edge_start], edge_idx_2_nodes[edge_end]
    node_start, node_end = edge_nodes_start[0], edge_nodes_end[0]
    path_nodes = nx.dijkstra_path(G, node_start, node_end, weight='weight')

    path_edges = []
    for i in range(len(path_nodes) - 1):
        path_edges.append(nodes_2_edge_id[(path_nodes[i], path_nodes[i + 1])])

    coords = []
    for edge_id in path_edges:
        coords.append(edge_idx_2_points[edge_id])
    coords = [coord for line in coords for coord in line]
    coords = [{'lat': coords[i][0], 'lng': coords[i][1]} for i in range(len(coords))]

    return path_edges, coords
