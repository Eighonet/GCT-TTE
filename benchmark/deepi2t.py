from torch import nn 
import torch
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU



class ResLinearBlock(nn.Module):
    def __init__(self):
        super(ResLinearBlock, self).__init__()
        self.linear_1 = nn.Linear(2*512, 2*512, bias=False)
        self.linear_2 = nn.Linear(2*512, 2*512, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(2*512)
        self.batch_norm_2 = nn.BatchNorm1d(2*512)

    def forward(self, x):
        result = self.linear_1(x)
        result = F.relu(self.batch_norm_1(result))
        result = self.linear_2(result)
        result = F.relu(self.batch_norm_2(result))

        return x + result

    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2d(3, 8, (3, 3))
        self.pool1 = MaxPool2d((2, 2), stride=2)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = Conv2d(8, 16, (3, 3))
        self.pool2 = MaxPool2d((3, 3), stride=3)
        self.relu2 = ReLU(inplace=True)
        self.conv3 = Conv2d(16, 8, (3, 3))
        self.pool3 = MaxPool2d((2, 2), stride=3)
        self.relu3 = ReLU(inplace=True)
        self.fc = Linear(8 * 13 * 13, 200)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = self.fc(torch.flatten(x, 1))
        return x


def get_images(idxs_batch, device):
    batch_images = []
    for idxs in idxs_batch:
        images = []
        for idx in idxs:
            name = CROPS_PATH + image_cors[idx]['patch_id_1']
            my_img = cv2.imread(name)
            my_img = np.transpose(my_img, (2, 0, 1))
            my_img_tensor = torch.from_numpy(my_img)
            my_img_tensor = my_img_tensor.type("torch.FloatTensor")
            my_img_tensor *= 1 / 255
            my_img_tensor = my_img_tensor.unsqueeze(0)
            my_img_tensor.to(device)
            images.append(my_img_tensor)
        batch_images.append(
            torch.stack(images).squeeze(1).type("torch.FloatTensor").to(device)
        )
    return batch_images


def get_node_embeddings(nodes_batch, device):
    batch_embs = []
    for nodes in nodes_batch:
        embs = []
        nodes = [str(i) for i in nodes]
        for node in nodes:
            embs.append(node_embeddings[node])
        if len(embs) > 1:
            batch_embs.append(
                torch.stack(
                    [torch.from_numpy(item).float().to(device) for item in embs]
                ).squeeze(-2)
            )
        else:
            batch_embs.append(
                torch.stack(
                    [torch.from_numpy(item).float().to(device) for item in embs]
                )
                .squeeze(-2)
                .unsqueeze(dim=0)
            )
    return batch_embs


def get_flow_embeddings(nodes_batch, device):
    batch_embs = []
    for nodes in nodes_batch:
        embs = []
        nodes = [i for i in nodes]
        for node in nodes:
            embs.append(flow_emb[node])
        if len(embs) > 1:
            batch_embs.append(
                torch.stack(
                    [torch.from_numpy(item).float().to(device) for item in embs]
                ).squeeze(-2)
            )
        else:
            batch_embs.append(
                torch.stack(
                    [torch.from_numpy(item).float().to(device) for item in embs]
                )
                .squeeze(-2)
                .unsqueeze(dim=0)
            )
    return batch_embs


direction_map = {
    "WN": 0,
    "WE": 1,
    "WS": 2,
    "NE": 3,
    "NS": 4,
    "NW": 5,
    "ES": 6,
    "EW": 7,
    "EN": 8,
    "SW": 9,
    "SN": 10,
    "SE": 11,
    "WW": 12,
    "EE": 13,
    "SS": 14,
    "NN": 15
}


def get_sign(x, y):
    d = y - x
    d_alpha = np.arctan2(d[1], d[0]) * ALPHA
    if -45 < d_alpha <= 45:
        return "E"
    elif 45 < d_alpha <= 135:
        return "N"
    elif -135 < d_alpha <= -45:
        return "S"
    return "W"


def get_directions(image_idxs_batch, geo_batch):
    batch_directions = []
    for img_idxs, geo in zip(image_idxs_batch, geo_batch):
        directions = []
        seq_len = len(img_idxs) - 1
        img_values = str_to_list_floats(image_cors[img_idxs[0]]['coord_id_1'])
        prev = get_sign(np.array(img_values), geo[0])
        for idx in range(seq_len):
            cur = get_sign(geo[idx], geo[idx + 1])
            directions.append(direction_map[prev + cur])
            prev = cur
        img_values = str_to_list_floats(image_cors[img_idxs[seq_len]]['coord_id_1'])
        cur = get_sign(geo[seq_len], np.array(img_values))
        directions.append(direction_map[prev + cur])
        batch_directions.append(directions)
    return batch_directions


class DeepI2T(nn.Module):
    def __init__(self, device):
        super(DeepI2T, self).__init__()
        self.convnet = ConvNet()
        self.directions = nn.Embedding(16, 200)
        self.bilstm = nn.LSTM(input_lstm_dim, 512, bidirectional=True, batch_first=True)
        self.res_block1 = ResLinearBlock()
        self.res_block2 = ResLinearBlock()
        self.res_block3 = ResLinearBlock()
        self.fc = nn.Linear(2*512, 1)
        self.device = device
        self.num_layers = 1

    def forward(self, features, nodes, geo):
        features = torch.tensor(features).to(self.device)
        if with_line:
            ne = get_node_embeddings(nodes)
        if with_flow:
            fe = get_flow_embeddings(nodes)
        if with_conv:
            ie = [self.convnet(img) for img in get_images(nodes)]
        if with_direction and with_conv:
            directions = [self.directions(torch.tensor(direct).to(self.device)) for direct in get_directions(nodes, geo)]
            ie = [torch.mul(img, direct) for img, direct in zip(ie, directions)]
        if not with_line:
            lens = [min(len(img), len(flw)) for img, flw in zip(ie, fe)]
            embs = [torch.cat((img[:len_i, :], flw[:len_i, :]), dim=-1) for len_i, img, flw in zip(lens, ie, fe)]
        elif not with_conv:
            lens = [min(len(nodes), len(flw)) for nodes, flw in zip(ne, fe)]
            embs = [torch.cat((nodes[:len_i, :], flw[:len_i, :]), dim=-1) for len_i, nodes, flw in zip(lens, ne, fe)]
        elif not with_flow:
            lens = [min(len(img), len(nodes)) for img, nodes in zip(ie, ne)]
            embs = [torch.cat((img[:len_i, :], nodes[:len_i, :]), dim=-1) for len_i, img, nodes in zip(lens, ie, ne)]
        else:
            lens = [min(len(img), len(nodes), len(flw)) for img, nodes, flw in zip(ie, ne, fe)]
            embs = [torch.cat((img[:len_i, :], nodes[:len_i, :], flw[:len_i, :]), dim=-1)
                    for len_i, img, nodes, flw in zip(lens, ie, ne, fe)]
        features = [
            f.unsqueeze(-2).expand(e.shape[0], -1) for f, e in zip(features, embs)
        ]
        embs = [
            torch.cat((e, f), dim=-1).unsqueeze(1).float().to(self.device)
            for e, f in zip(embs, features)
        ]
        h_outs = []
        for e in embs:
            h_0 = Variable(torch.zeros(2 * self.num_layers, 1, 512)).to(
                self.device
            )
            c_0 = Variable(torch.zeros(2 * self.num_layers, 1, 512)).to(
                self.device
            )
            self.bilstm.flatten_parameters()
            ula, (h_out, _) = self.bilstm(torch.transpose(e, 0, 1), (h_0, c_0))
            del h_out
            torch.cuda.empty_cache()
            ula = ula.view(-1, 1024).mean(0)
            h_outs.append(ula)
        h_outs = torch.stack(h_outs)

        x = self.res_block1(h_outs)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = F.relu(self.fc(x))

        return x
    

if __name__ == "__main__":
    device = "cpu"
    input_lstm_dim = 338
    
    deepi2t = DeepI2T(device)
    deepi2t.load_state_dict(torch.load('trained_model.pth'))