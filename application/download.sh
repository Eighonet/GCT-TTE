#!/bin/sh
mkdir data
mkdir inference/inference_data
cd data
wget -O omsk_aux_data.npy https://sc.link/nE54
wget -O abakan_aux_data.npy https://sc.link/oG2z
wget -O abakan.npy https://sc.link/pJY1
wget -O omsk.npy https://sc.link/qKx3
wget -O omsk_weighted.gpickle https://sc.link/rLkw
wget -O abakan_weighted.gpickle https://sc.link/vP2V
cd ../inference/inference_data
wget -O O_nodes_features.csv https://sc.link/wQpJ
wget -O OmskInfNoImgs.pt https://sc.link/xRkJ
wget -O O_edge_index.npz https://sc.link/BwBx
wget -O AbakanInfNoImgs.pt https://sc.link/AvDB
wget -O A_nodes_features.csv https://sc.link/zWpr
wget -O A_edge_index.npz https://sc.link/yVPV
