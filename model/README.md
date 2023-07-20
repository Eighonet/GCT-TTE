# Launch Instructions

You can use the following commands to control the model settings:

- `-e` -- number of epochs.
- `-lr` -- learning rate.
- `--optimizer-name` -- optimizer name from `torch.optim`.
- `-s` -- the index of split, dividing data to train and test,  from applied train dataset.
- `-bs` -- batch size.
- `-nd` -- device number, i.e. for "cuda:0" it is 0.
- `-o` -- logs saving directory 
- `--alpha` --  coefficient of graph impact (from $0$ to $1$, for images it will be $1 - \alpha$)
- `--alpha-feat` -- it is $\beta$ impact coefficient for auxiliary features priority.
- `--path-blind` -- forces the path-blind mode for model if it is set to `True`, otherwise (`False`) will turn on the path-aware mode.
- `--kfold-filename` -- name of splits' file needed to use, specified in dataset description section.
- `--city`-- name of the city: "Abakan" or "Omsk".
- `--graph-layers` -- number of graph convolution layers.
- `--hidden-size` -- the output  size of RegNet and GCN layers. If it is set to $n$ then the input for transformer encoder will be $2n$ $+$ size of auxiliary features.
- `--linear-size`  -- size of auxiliary features vector.
- `--encoder-layers`  -- number of transformer encoder layers.
- `--fuse-layers` -- number of fine-tune layers for regression task.
- `--seq-len` -- the fixed length of transformer sequence. Each trip will be truncated or padded regarding this parameter.
- `--graph-input-size`  -- the size of input vector for the graph convolution layers.
- `--num-heads` -- number of attention heads in transformer.
- `--use-infomax` -- if it is set to 1/0 then deep graph infomax will be used/not used. Implementation: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/deep_graph_infomax.html 

### Quick test quide

1) Please, install PyTorch and PyTorch-Geometric in your environment;
2) unzip the dataset;
3) unzip the archive with [pretrained GCT-TTE weights](https://sc.link/xnl8z); Ask _**semenova.bnl@gmail.com**_ if you will have any troubles with downloading.
4) check the paths' constants in `test.py`;
5) launch test sctipt as `python test.py`, you can choose the training k-fold split via `-s <n>` option, where `n` is the split number.

The metrics will appear after test prrocess bieng finished.

Example for launching on Abakan:

`python test.py --city Abakan --graph-input-size 73 -s 0 --batch-size 16`

Example for launching on Omsk:

`python test.py --city Omsk -o Omsk --graph-input-size 100 -s 0 --batch-size 16`

# GCT-TTE dataset description

To get the dataset, please contact the correspondent author at this email address _**semenova.bnl@gmail.com**_

## Training dataset

Training dataset consists of two parts for Abakan and Omsk cities. You need to use this functions to unpack correctly some files:

```python
def stringToIntList(string: str) -> list:
    lst = [int(s) for s in np.array(re.sub("[\[,'\]]", '', string).split(' '))]
    return np.array(lst)

def stringToStrList(string: str) -> np.array:
    return np.array(re.sub("[\[,'\]\\n]", '', string).split(' '))

def stringToFloatList(string: str) -> list:
    lst = [float(s) for s in np.array(re.sub("[\[,'\]]", '', string).split(' '))]
    return np.array(lst)
```

To deal with version conflicts, we tried to avoid using of `pickle` module wherever it is possible.

### Abakan data

The dataset is compressed into `ABAKAN_TRAIN_DATA.tar.gz`:

- `IMG_EMBS.npz`
- `edge_index.npz`
- `extra_features.csv`
- `indexes.pkl`
- `nodes_features.csv`
- `route_2_Tids.csv`
- `route_2_xIDs.csv`
- `targets.csv`

Here is the description of the dataset files:

- `nodes_features.csv` is a .csv table that contains all the nodes' $\Re^{73}$ vector representations of road city graph.  The table has a size of $65524$ rows and $73$ columns. 

- `targets.csv` is a .csv table containing $121557$ rows for keeping the target value, the travel time of a trip. 

- `edge_index.npz` is NumPy zip file containing the edge list of the road city graph, connections between roads.

- `IMG_EMBS.npz` is a NumPy zip file containing embeddings of RegNetY model in a tensor of size $\left[13802, 3712\right]$, where $3712$ is an embedding size and $13802$ is a number of all processed grid-based images.

- `route_2_Tids.csv` is a mapping from trip ID to a sequence of IDs in `IMG_EMBS.npz` tensor. It is used to make an image representation tensor of a trip.

- `route_2_xIDs.csv` is a mapping from trip ID to its nodes' representations from `nodes_features.csv`.

- `extra_features.csv` is a .csv table $121557$ rows $\times$ $10$ columns. Each row is a vector of time and weather conditions for a trip. The row ID and the trip ID are the same. The list of features: clouds, snow, temperature, wind direction, wind speed,  pressure, dateID, week period, weekID, timeID.
  
  - clouds -- from $0$ (clear) to $10$ (extremely cloudy).
  - snow -- from $0$ (clear) to $2$ (snowy).
  - temperature -- from $-31.0$ to $13.0$ Celsius degrees.
  - wind direction -- from $0$ to $315$ degrees.
  - wind speed -- from $0$ to $13$ meters per second.
  - pressure -- from $736.0$ to $764.0$ millimeters of mercury.
  - dateID -- the day of the month. From $0$ to $31$.
  - week period -- if the trip happened in the weekends it is set to $1$, otherwise it is $0$.
  - weekID -- for Monday it is $0$, for Tuesday it is $1$, for Wednesday it is $2$ etc.
  - timeID --  number of minutes since the beginning of the day.  

- `indexes.pkl` is the only pickle file in the dataset. It is serialized using the $3$'rd pickle protocol. It contains a python dictionary with five k-fold splits, we used in training. Each split contains 'train' and 'valid' arrays of trips' indices. The dictionary tree can be represented as:
  
  ```python
  {0: {'train': array([     1,      2,      3, ...]),
  'valid': array([     4,      5,     6, ...])},
   ...
  {4: {'train': array([     1,      2,      3, ...]),
  'valid': array([     4,      5,     6, ...])},
  ```

### Omsk data

The dataset is compressed into `OMSK_TRAIN_DATA.tar.gz`:

- `IMG_EMBS.npz`
- `edge_index.npz`
- `extra_features.csv`
- `indexes.pkl`
- `nodes_features.csv`
- `route_2_Tids.csv`
- `route_2_xIDs.csv`
- `targets.csv`

Omsk contains information about $767343$ trips and $231688$ roads, the files' organization is the same as one of Abakan. The difference is only in the following moment:

- The road vector representation is  $\Re^{100}$ vector, so `nodes_features.csv` is table of size $231688$ $\times$ $100$.

## Inference dataset

<table style="undefined;table-layout: fixed; width: 599px">
<colgroup>
<col style="width: 169px">
<col style="width: 323px">
<col style="width: 107px">
</colgroup>
<thead>
  <tr>
    <th>Filename</th>
    <th>Description</th>
    <th>Shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>AbakanInfNoImgs.pt</td>
    <td>The weights of Abakan GCT-TTE inference model.<br></td>
    <td>-</td>
  </tr>
  <tr>
    <td>OmskInfNoImgs.pt</td>
    <td>The weights of Omsk GCT-TTE inference model.<br></td>
    <td>-</td>
  </tr>
  <tr>
    <td>A_edge_index.npz</td>
    <td>It is NumPy zip file containing the edge list of the road city graph, connections between roads. It is a copy of edge_index.npz from Abakan train dataset.</td>
    <td>(2, 340012)</td>
  </tr>
  <tr>
    <td>O_edge_index.npz</td>
    <td>As A_edge_index.npz but for Omsk. It is a copy of edge_index.npz from Omsk train dataset.</td>
    <td>(2, 1149492)</td>
  </tr>
  <tr>
    <td>A_nodes_features.csv</td>
    <td>It is a .csv table that contains all the node embeddings of road city graph relatively to 2GIS road map. It is a copy of nodes_features.csv from Abakan train dataset.</td>
    <td>(65524, 73)</td>
  </tr>
  <tr>
    <td>O_nodes_features.csv</td>
    <td>As A_nodes_features.csv but for Omsk. It is a copy of nodes_features.csv from Omsk train dataset.</td>
    <td>(231688, 100)</td>
  </tr>
</tbody>
</table>
