import torch

X_ABAKAN_PATH = './inference_data/A_nodes_features.csv'
X_OMSK_PATH = './inference_data/O_nodes_features.csv'
EDGE_INDEX_ABAKAN = './inference_data/A_edge_index.npz'
EDGE_INDEX_OMSK = './inference_data/O_edge_index.npz'
ABAKAN_STATE_DICT = './inference_data/AbakanInfNoImgs.pt'
OMSK_STATE_DICT = './inference_data/OmskInfNoImgs.pt'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

PADDING_VALUE = 0
SEED = 1234
N_HEADS = 4
N_IENC_LAYERS = 2
N_GENC_LAYERS = 2
N_DEC_LAYERS = 2

PATH_BLIND = False

ALPHA = 1.0
TRANSFORMER_ENCODER_LAYERS = 2
HIDDEN_SIZE = 160
GRAPH_LAYERS = 3
LINEAR_SIZE = 32

SEQ_LEN_ABAKAN = 128
SEQ_LEN_OMSK = 200

GRAPH_FEAT_SIZE_ABAKAN = 73
GRAPH_FEAT_SIZE_OMSK = 100

# === API ==========

graph_hidden_size = HIDDEN_SIZE
n_graph_layers = GRAPH_LAYERS
graph_model = "GCNConv"
use_infomax = 1
n_encoder_layers = TRANSFORMER_ENCODER_LAYERS
n_fuse_layers = 3
alpha_g = ALPHA
alpha_feat=0.1

EXTRAS_ABAKAN = torch.tensor([[-0.4322, -0.2236,  1.1487,  0.2966,  0.5759,  0.0157, -0.0580, -0.6552, 0.0402,  0.0820]], 
                             dtype=torch.float, device=device)
EXTRAS_OMSK = torch.tensor([[-0.4374, -0.2319,  1.2225,  0.2513, -0.3780, -0.2536,  0.0932, -0.6534, 0.5612,  0.1287]], 
                           dtype=torch.float, device=device)
