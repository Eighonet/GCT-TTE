from typing import Tuple
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric
from igraph import *
import pickle


def get_tree(clear_nodes):
    cors = []
    for i in range(len(clear_nodes)):
        cors.append([float(clear_nodes[i]["lat"]), float(clear_nodes[i]["lon"])])

    X = np.array([[np.radians(x) for x in list_] for list_ in cors])
    tree = BallTree(X, leaf_size=3, metric=DistanceMetric.get_metric('haversine'))

    return tree


# path_to_g = dijkstra.pickle
# path_to_nodes = clear_nodes.pkl
class DijkstraPath:
    def __init__(self, path_to_g, path_to_nodes):
        self.g = pickle.load(open(path_to_g, 'rb'))
        self.nodes = pickle.load(open(path_to_nodes, 'rb'))
        self.tree = get_tree(self.nodes)

    def get_shortest_path(self, cor1: Tuple[float, float], cor2: Tuple[float, float], weights):
        _, idx1 = self.tree.query([[cor1[0] * np.pi / 180, cor1[1] * np.pi / 180]], k=1)
        _, idx2 = self.tree.query([[cor2[0] * np.pi / 180, cor2[1] * np.pi / 180]], k=1)

        path = self.g.get_shortest_paths(idx1[0][0], to=idx2[0][0], mode=OUT, weights=weights)[0]

        cor_path = [[float(self.nodes[idx1[0][0]]['lat']), float(self.nodes[idx1[0][0]]['lon'])]]
        for i in range(len(path)):
            cor_path.append([float(self.nodes[path[i]]['lat']), float(self.nodes[path[i]]['lon'])])
        cor_path.append([float(self.nodes[idx1[0][0]]['lat']), float(self.nodes[idx1[0][0]]['lon'])])

        edge_ids = []
        for i in range(len(path)-1):
            edge_ids.append(self.g.get_eid(path[i], path[i + 1]))
            # edge_ids.append(int(self.nodes[path[i]]['id']))
        return edge_ids, cor_path

def get_eta_determin(edges):
    time_ = 0
    for edge_id in edges:
        dist_ = node_data.iloc[edge_id].dist
        speed = average_speed.get(edge_id, 15)
        if speed <= 0:
            speed = 1
        time_ += dist_ / speed
    return time_


if __name__ == "__main__":
    BASE = Path(os.path.realpath(__file__)).parent

    m = DijkstraPath(BASE / 'data/dijkstra.pickle', BASE / 'data/clear_nodes.pkl')
    with open(BASE / 'data/weights/hist.pkl', 'rb') as handle:
        weights = pickle.load(handle)
    result = m.get_shortest_path((55.759669, 37.573474), (55.769550, 37.565640), weights)
    print(result[0])
