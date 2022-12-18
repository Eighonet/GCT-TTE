import os
import argparse
import statistics as stat
import random
import pickle as pk
from tqdm.autonotebook import tqdm

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from gct_tte import GCTTTE
from utils import stringToIntList, stringToStrList, stringToFloatList
from utils import get_padding_mask, corruption, pad_seq_fn, make_positional_encoding
from utils import train, test
from dataset import DataSet
from configs import *


parser = argparse.ArgumentParser(description="model parameters")
parser.add_argument("-e", "--epoch", type=int, default=100, help="number of epochs")
parser.add_argument("-lr", type=float, default=0.00005234392263346623, help="learning rate")
parser.add_argument("--optimizer-name", type=str, default="Adam",
                    help="group name for wandb")
parser.add_argument("-s", "--split-num-from-kfold", type=int, default=0, help="split number")
parser.add_argument("-bs", "--batch-size", type=int, default=16, help="batch size on train")
parser.add_argument("-nd", "--num-device", type=int, default=0, help="index of device")
parser.add_argument("-o", "--output-path", type=str, default="MTTE_Abakan_ratio",
                    help="path to output derictory")
parser.add_argument("--alpha", type=float, default=1.0,
                    help="the ratio of graph and image influence")
parser.add_argument("--alpha-feat", type=float, default=0.1,
                    help="coefficient for features")
parser.add_argument("--path-blind", default=True, action="store_false",
                    help="if we had only start and finish point")
parser.add_argument("--kfold-filename", type=str, default="indexes_k8",
                    help="group name for wandb")
parser.add_argument("--city", type=str, default="Abakan", help="city")
parser.add_argument("--graph-layers", type=int, default=3, help="number of graph convolutional layers")
parser.add_argument("--hidden-size", type=int, default=160, help="hidden size")
parser.add_argument("--linear-size", type=int, default=32, help="linear layer size for features")
parser.add_argument("--encoder-layers", type=int, default=2, help="number of Transformer Encoder layers")
parser.add_argument("--fuse-layers", type=int, default=3, help="number of fuse layers")
parser.add_argument("--seq-len", type=int, default=128, help="sequence length")
parser.add_argument("--graph-input-size", type=int, default=73, help="graph input size")
parser.add_argument("--seed", type=int, default=1234, help="seed")
parser.add_argument("--num-heads", type=int, default=4, help="number of heads of attention")
parser.add_argument("--use-infomax", type=int, default=1, help="if use infomax 1, 0 else")


args = parser.parse_args()
EPOCHS = args.epoch
LR = args.lr
OPTIMIZER_NAME = args.optimizer_name
N_SPLIT = args.split_num_from_kfold
BATCH_SIZE = args.batch_size
DEVICE = torch.device("cuda:{}".format(args.num_device) if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = args.output_path
GROUP_NAME = args.group_name
ALPHA = args.alpha
ALPHA_FEAT = args.alpha_feat
PATH_BLIND = not args.path_blind
KFOLD_FILENAME = args.kfold_filename
CITY = args.city
GRAPH_LAYERS = args.graph_layers
HIDDEN_SIZE = args.hidden_size
LINEAR_SIZE = args.linear_size
TRANSFORMER_ENCODER_LAYERS = args.encoder_layers
FUSE_LAYERS = args.fuse_layers
SEQ_LEN = args.seq_len
GRAPH_INPUT_SIZE = args.graph_input_size
SEED = args.seed
N_HEADS = args.num_heads
USE_INFOMAX = args.use_infomax
N_WORKERS=8


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


try:
    os.makedirs('output/{}/{}'.format(OUTPUT_PATH, N_SPLIT))
except:
    pass


# load data
if CITY == "Abakan":
    print("City is Abakan")
    print("Getting x data")
    x = pd.read_csv(os.path.join(PATH_ABAKAN, "nodes_features.csv")).to_numpy()

    print("Getting y data")
    y = pd.read_csv(os.path.join(PATH_ABAKAN, "targets.csv")).to_numpy()

    print("Getting edge_index data")
    edge_index = np.load(open(os.path.join(PATH_ABAKAN, "edge_index.npz"), 'rb')) # Возможно, он тут не нужен

    print("Getting route_Tids data")
    route_Tids = pd.read_csv(os.path.join(PATH_ABAKAN, "route_2_Tids.csv"))['0']
    route_Tids = pd.Series([stringToIntList(ids[1]) for ids in route_Tids.iteritems()])

    print("Getting route_2_nodeIdSeq data")
    route_2_nodeIdSeq = pd.read_csv(os.path.join(PATH_ABAKAN, "route_2_xIDs.csv"))['0']
    route_2_nodeIdSeq = pd.Series([stringToIntList(ids[1]) for ids in route_2_nodeIdSeq.iteritems()])

    print("Getting train_route_id_2_list data")
    train_route_id_2_list = pk.load(open(os.path.join(PATH_ABAKAN, "{}.pkl".format(KFOLD_FILENAME)),
                                         'rb')
                                   )

    print("Getting extra_features data")
    extra_features = pd.read_csv(os.path.join(PATH_ABAKAN, "extra_features.csv")).to_numpy()
    
    print("Getting images") # Это уже готовые эмбеддинги
    IMG_EMBS = os.path.join(PATH_ABAKAN, "IMG_EMBS.npz")
    TENSORS = np.load(IMG_EMBS)
    TENSORS = torch.tensor(TENSORS, device=torch.device('cpu'))

    
elif CITY == "Omsk":
    print("City is Omsk")
    print("Getting x data")
    x = pd.read_csv(os.path.join(PATH_OMSK, "nodes_features.csv")).to_numpy()

    print("Getting y data")
    y = pd.read_csv(os.path.join(PATH_OMSK, "targets.csv")).to_numpy()

    print("Getting edge_index data")
    edge_index = np.load(open(os.path.join(PATH_OMSK, "edge_index.npz"), 'rb')) # Возможно, он тут не нужен

    print("Getting route_Tids data")
    route_Tids = pd.read_csv(os.path.join(PATH_OMSK, "route_2_Tids.csv"))['0']
    route_Tids = pd.Series([stringToIntList(ids[1]) for ids in route_Tids.iteritems()])

    print("Getting route_2_nodeIdSeq data")
    route_2_nodeIdSeq = pd.read_csv(os.path.join(PATH_OMSK, "route_2_xIDs.csv"))['0']
    route_2_nodeIdSeq = pd.Series([stringToIntList(ids[1]) for ids in route_2_nodeIdSeq.iteritems()])

    print("Getting train_route_id_2_list data")
    #train_route_id_2_list = pk.load(open('/home/jovyan/MTTE/Omsk_data/indexes_f.pkl', 'rb'))
    train_route_id_2_list = pk.load(open(os.path.join(PATH_OMSK, "indexes.pkl"), 'rb'))

    print("Getting extra_features data")
    extra_features = pd.read_csv(os.path.join(PATH_OMSK, "extra_features.csv")).to_numpy()

    with open(os.path.join(PATH_OMSK, "omsk_full_routes_deeptte_states.pkl"), 'rb') as f:
        X = pk.load(f)

    X.state = pd.factorize(X.state)[0]
    X.index = list(range(X.shape[0]))

    neg_index = []
    for i, line in enumerate(X.edges_int):
        if len(line) < 2:
            neg_index.append(i)

    X.drop(neg_index, inplace=True)
    X = X[(X["RTA"] > 30) & (X["RTA"] < 3000)]

    new_mapping = {ind:i for i, ind in enumerate(X.index)}

    for num_split in train_route_id_2_list:
        res_train = []
        res_valid = []
        for i in train_route_id_2_list[num_split]['train']:
            try:
                res_train.append(new_mapping[i])
            except:
                continue
        for i in train_route_id_2_list[num_split]['valid']:
            try:
                res_valid.append(new_mapping[i])
            except:
                continue
        train_route_id_2_list[num_split]['train'] = res_train
        train_route_id_2_list[num_split]['valid'] = res_valid
        
    print("Getting images")
    TENSORS = torch.zeros(47447, 3712)
    

data = Data(
    x=torch.tensor(x, dtype=torch.float), # признаковое описание
    edge_index=torch.tensor(edge_index, dtype=torch.long), # транспонированный список ребер
)
data = data.to(DEVICE)

scaler = StandardScaler()
scaler.fit(extra_features[train_route_id_2_list[N_SPLIT]['train']])
extra_features = scaler.transform(extra_features)

train_dataset = DataSet(targets=y, iter_2_id=np.array(train_route_id_2_list[N_SPLIT]['train']), 
                        route_2_nodeIdSeq=route_2_nodeIdSeq, route_2_Tids=route_Tids, extra_features=extra_features,
                        sequence_length=SEQ_LEN, path_blind=PATH_BLIND)

test_dataset = DataSet(targets=y, iter_2_id=np.array(train_route_id_2_list[N_SPLIT]['valid']), 
                      route_2_nodeIdSeq=route_2_nodeIdSeq, route_2_Tids=route_Tids, extra_features=extra_features,
                      sequence_length=SEQ_LEN, path_blind=PATH_BLIND)

# load model
model = GCTTTE(data,
               model_hidden_size=HIDDEN_SIZE,
               n_graph_layers=GRAPH_LAYERS,
               use_infomax=USE_INFOMAX,
               graph_model_name="GCNConv",
               num_transf_enc_layers=TRANSFORMER_ENCODER_LAYERS,
               n_out_layers=FUSE_LAYERS,
               alpha_g=ALPHA,
               alpha_feat=ALPHA_FEAT,
               linear_size=LINEAR_SIZE,
               n_heads=N_HEADS,
               img_tensors=TENSORS,
               graph_input_size=GRAPH_INPUT_SIZE,
               device=DEVICE,
               path_plind=PATH_BLIND,
               seq_len=SEQ_LEN,
              ).to(DEVICE)

PATH = 'output/{}/{}/best_mae_model_alpha_{}_pathblind_{}_trEn{}_hid{}_graphL{}.pt'.format(
                            OUTPUT_PATH, N_SPLIT, ALPHA, PATH_BLIND, TRANSFORMER_ENCODER_LAYERS, HIDDEN_SIZE, GRAPH_LAYERS)
model.load_state_dict(torch.load(PATH))
model.eval()
optimizer = getattr(torch.optim, OPTIMIZER_NAME)(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


def test_run():
    temp_dict = {}
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=N_WORKERS, shuffle=True)
    val_loss_mape = []
    val_loss_mae = []
    val_loss_sr = []
    val_loss_rmse = []

    for index, (graph_embed, labels, imgs_embed, extras, lengths) in enumerate(tqdm(test_dataloader)):
        l = test(model, DEVICE, data, graph_embed, labels, imgs_embed, extras, lengths)

        val_loss_mape.append(l[0])
        val_loss_mae.append(l[1])
        val_loss_sr.append(l[2])
        val_loss_rmse.append(l[3])

    temp_dict["test/MAPE"] = epoch_test_loss_mape
    temp_dict["test/MAE"] =  epoch_test_loss_mae
    temp_dict["test/SR"] = epoch_test_loss_sr
    temp_dict["test/RMSE"] = epoch_test_loss_rmse

    with open('output/{}/{}/test_{}_pathblind_{}_trEn{}_hid{}_graphL{}.pkl'.format(
        OUTPUT_PATH, N_SPLIT, ALPHA, PATH_BLIND, TRANSFORMER_ENCODER_LAYERS, HIDDEN_SIZE, GRAPH_LAYERS), 'wb') as f:
        pk.dump(temp_dict, f)
    return stat.mean(val_loss_mape)

        
if __name__ == '__main__':
    test_run()
