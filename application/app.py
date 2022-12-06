import random

from flask import Flask, jsonify, request, render_template
import numpy as np
import networkx as nx
from scipy import spatial

from utils import get_route
from inference.API import GCT_TTE_inference

G = {'abakan':nx.read_gpickle("data/abakan_weighted.gpickle"),\
     'omsk':nx.read_gpickle("data/omsk_weighted.gpickle")}

aux_data = {'abakan':np.load('data/abakan_aux_data.npy', allow_pickle=True).item(),\
            'omsk':np.load('data/omsk_aux_data.npy', allow_pickle=True).item()}

kd_tree = {'abakan':spatial.KDTree(aux_data['abakan']['available_points']),\
            'omsk':spatial.KDTree(aux_data['omsk']['available_points'])}

predefined_data = {'abakan':np.load('data/abakan.npy', allow_pickle=True),\
                   'omsk':np.load('data/omsk.npy', allow_pickle=True)}

app = Flask(__name__)

model = GCT_TTE_inference()

@app.route('/')
def init():
    return render_template('index.html')

@app.route('/getdata/', methods=['GET', 'POST'])
def data_get():
    get_coords = lambda x: [float(coord) for coord in x[0].split(',')]

    data = request.form.to_dict(flat=False)
    start_point, end_point = get_coords(data['start_point']), get_coords(data['end_point'])
    city = data['city'][0]

    edges, coords = get_route(start_point, end_point, kd_tree[city],\
                              G[city], aux_data[city]['nodes_2_edge_id'], aux_data[city]['point_2_edge'],\
                              aux_data[city]['edge_idx_2_nodes'], aux_data[city]['edge_idx_2_points'])

    tte = model.predict(city, edges)

    return jsonify(
        tte=tte,
        coords=coords
        )

@app.route('/predefined_data/', methods=['GET', 'POST'])
def predefined_get():
    data = request.form.to_dict(flat=False)

    city = data['city'][0]
    routes_number = 3
    print(list(np.random.choice(predefined_data[city], routes_number)))
    return jsonify(
        predefined_routes=random.sample(list(predefined_data[city]), routes_number)
        )

if __name__ == '__main__':
    app.run()
