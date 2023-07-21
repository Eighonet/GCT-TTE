from torch import nn 
import torch
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU


class PathCNN(torch.nn.Module):
    def __init__(self):
        super(PathCNN, self).__init__()
        # Conv Level 1
        self.conv1max = torch.nn.Conv2d(3, 16, 3, padding=True)
        self.conv1avg = torch.nn.Conv2d(3, 16, 3, padding=True)
        self.maxPool1 = torch.nn.MaxPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.avgPool1 = torch.nn.AvgPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_features=16)
        # Conv Level 2
        self.conv2max = torch.nn.Conv2d(32, 32, 3, padding=True)
        self.conv2avg = torch.nn.Conv2d(32, 32, 3, padding=True)
        self.maxPool2 = torch.nn.MaxPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.avgPool2 = torch.nn.AvgPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.batchNorm2 = torch.nn.BatchNorm2d(num_features=32)
        # Conv Level 3
        self.conv3max = torch.nn.Conv2d(64, 64, 3, padding=True)
        self.conv3avg = torch.nn.Conv2d(64, 64, 3, padding=True)
        self.maxPool3 = torch.nn.MaxPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.avgPool3 = torch.nn.AvgPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.batchNorm3 = torch.nn.BatchNorm2d(num_features=64)
        # Conv Level 4
        self.conv4max = torch.nn.Conv2d(128, 128, 3, padding=True)
        self.conv4avg = torch.nn.Conv2d(128, 128, 3, padding=True)
        self.maxPool4 = torch.nn.MaxPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.avgPool4 = torch.nn.AvgPool2d(torch.Size([2, 2]), stride=[2, 2], ceil_mode=True)
        self.batchNorm4 = torch.nn.BatchNorm2d(num_features=128)
        # Flattening
        self.flatten = torch.nn.Flatten()
        # Fully connected
        self.dropout = torch.nn.Dropout(p=0.1)
        self.fc = torch.nn.Linear(65536, 1024)

    def forward(self, x):
        # Conv Level 1
        x1 = self.batchNorm1(self.maxPool1(F.relu(self.conv1max(x))))
        x2 = self.batchNorm1(self.avgPool1(F.relu(self.conv1avg(x))))
        x = torch.cat([x1, x2], axis=1)
        # Conv Level 2
        x1 = self.batchNorm2(self.maxPool2(F.relu(self.conv2max(x))))
        x2 = self.batchNorm2(self.avgPool2(F.relu(self.conv2avg(x))))
        x = torch.cat([x1, x2], axis=1)
        # Conv Level 3
        x1 = self.batchNorm3(self.maxPool3(F.relu(self.conv3max(x))))
        x2 = self.batchNorm3(self.avgPool3(F.relu(self.conv3avg(x))))
        x = torch.cat([x1, x2], axis=1)
        # Conv Level 4
        x1 = self.batchNorm4(self.maxPool4(F.relu(self.conv4max(x))))
        x2 = self.batchNorm4(self.avgPool4(F.relu(self.conv4avg(x))))
        x = torch.cat([x1, x2], axis=1)
        # Flattening
        x = self.flatten(x)
        # FULLY CONNECTED
        x = self.fc(self.dropout(x)) # 0.1 ad hoc        
        return x
    

class TempLayer2(torch.nn.Module):
    
    def __init__(self):
        super(TempLayer2, self).__init__()
        self.batchNorm = torch.nn.BatchNorm1d(num_features=1024)
        # Conv Layer 1
        self.conv1 = torch.nn.Conv1d(1024, 1024, kernel_size=3, padding=True)
        self.maxPool1 = torch.nn.MaxPool1d(2, stride=2, ceil_mode=True)
        # Conv Layer 2
        self.conv2 = torch.nn.Conv1d(1024, 1024, kernel_size=3, padding=True)
        self.maxPool2 = torch.nn.MaxPool1d(2, stride=2, ceil_mode=True)
        # Conv Layer 3
        self.conv3 = torch.nn.Conv1d(1024, 1024, kernel_size=3, padding=True)
        self.maxPool3 = torch.nn.MaxPool1d(2, stride=2, ceil_mode=True)
        # Conv Layer 4
        self.conv4 = torch.nn.Conv1d(1024, 1024, kernel_size=3, padding=True)
        self.maxPool4 = torch.nn.MaxPool1d(2, stride=2, ceil_mode=True)
        # Flatten Layer
        self.flatten = torch.nn.Flatten()
        # Fully Connected Layer 1
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.fc1 = torch.nn.Linear(in_features=1024, out_features=1024)
        # Fully Connected Layer 2
        self.dropout2 = torch.nn.Dropout(p=0.1)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=1024)
        # Fully Connected Layer 3
        self.dropout3 = torch.nn.Dropout(p=0.1)
        self.fc3 = torch.nn.Linear(in_features=1024, out_features=1)
    
    def forward(self, x):
        x = self.batchNorm(x)
        # Conv Layer 1
        x = self.maxPool1(F.relu(self.conv1(x)))
        # Conv Layer 2
        x = self.maxPool2(F.relu(self.conv2(x)))
        # Conv Layer 3
        x = self.maxPool3(F.relu(self.conv3(x)))
        # Conv Layer 4
        if SEQ_LEN >= 9:
            x = self.maxPool4(F.relu(self.conv4(x)))
        # Flatten Layer
        x = self.flatten(x)
        # Fully Connected Layer 1
        # uncomment it if you have changed sequences len
        x = F.relu(self.fc1(self.dropout1(x)))
        # Fully Connected Layer 2
        x = F.relu(self.fc2(self.dropout2(x)))
        # Fully Connected Layer 3
        x = F.relu(self.fc3(self.dropout3(x)))
        return x
    
class DeepIST(torch.nn.Module):
    
    def __init__(self, max_seq_len, device):
        super(DeepIST, self).__init__()
        self.path_cnn = PathCNN()
        self.temp_layer = TempLayer2()
        self.max_seq_len = max_seq_len
        self.sigm = torch.nn.Softmax(-1)
        self.key_list = ['conv1max.weight',
                         'conv1avg.weight',
                         'conv2max.weight',
                         'conv2avg.weight',
                         'conv3max.weight',
                         'conv3avg.weight',
                         'conv4max.weight',
                         'conv4avg.weight',]
        self.without_center = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]]).to(device)
        self.ones_center2 = torch.zeros(3, 3).to(device)
        self.ones_center2[1, :] = 1
        self.ones_center2[:, 1] = 1
        
    def get_center_loss(self):
        return -torch.mean(torch.tensor(
                [torch.mean(self.path_cnn.state_dict()[k] * self.ones_center2) for k in self.key_list]
                ).to(device))

    def get_div_loss(self):
        res = []
        for k in self.key_list:
            temp_res = self.path_cnn.state_dict()[k] * self.without_center
            temp_res = self.sigm(temp_res[temp_res != 0].view(-1,4))
            res.append(torch.mean(- temp_res * torch.log(temp_res)))
        return -torch.mean(torch.tensor(res))
    
    def get_l2_loss(self):
        return torch.mean(
            torch.tensor([torch.mean(self.path_cnn.state_dict()[k] * self.path_cnn.state_dict()[k])
                          for k in self.key_list]))
    
    def get_loss(self, test: bool = False, g1=1.0, g2=1.0, g3=1.0):
        center_loss = self.get_center_loss()
        div_loss = self.get_div_loss()
        l2_loss = self.get_l2_loss()
        return abs(g1 * center_loss + g2 * div_loss + g3 * l2_loss)
    
    def forward(self, x):
        x = self.path_cnn(x)
        x = torch.unsqueeze(x.transpose(-1, -2), 0)
        # fix the size of the sequence
        x = torch.nn.functional.pad(x, (0, min(self.max_seq_len, self.max_seq_len - x.shape[-1])))
        return self.temp_layer(x)


    
if __name__ == "__main__":
    device = "cpu"
    seq_len = 15
    
    deepist = DeepIST(max_seq_len=seq_len, device=device)
    deepist.load_state_dict(torch.load('0/trained_model_mae.pth'))