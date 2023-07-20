from torch import nn 
import torch


class WDR(nn.Module):
    def __init__(
        self,
        recurrent_input_size,
        wide_input_size,
        deep_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        bidirectional,
        device,
    ):
        super(WDR, self).__init__()

        self.device = device

        # Recurrent part
        self.num_layers = lstm_num_layers
        self.hidden_size = lstm_hidden_size
        if bidirectional:
            self.bidirectional = 2
        else:
            self.bidirectional = 1

        self.linear_preprocess_lstm = nn.Linear(recurrent_input_size, 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Deep part
        self.linear_1 = nn.Linear(deep_input_size, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_2_1 = nn.Linear(256, 256)

        # Wide part
        self.linear_3 = nn.Linear(wide_input_size, 256)

        # Final regression
        self.linear_4 = nn.Linear(512 + 100, 256)
        self.linear_5 = nn.Linear(256, 1)

        # Test layers
        self.linear_test = nn.Linear(256, 1)
        self.branch_test = nn.Linear(deep_input_size, 256)

    def forward(self, deep_input, wide_input, indices):
        def merge_deep_branch(h_out, indices):
            deep_output = Variable(torch.Tensor(deep_input[indices])).to(self.device)
            deep_output = F.relu(self.linear_1(deep_output))
            deep_output = F.relu(self.linear_2(deep_output))
            deep_output = F.relu(self.linear_2_1(deep_output))
            return torch.cat((h_out, deep_output), 1)

        def merge_wide_branch(final_input, indices):
            wide_output = Variable(torch.Tensor(wide_input[indices])).to(self.device)
            wide_output = F.relu(self.linear_3(wide_output))
            return torch.cat((final_input, wide_output), 1)

        preprocessed_lstm_data = F.relu(
            self.linear_preprocess_lstm(
                pad_sequence([graph_features[i] for i in indices], batch_first=True).to(
                    self.device, dtype=torch.float
                )
            )
        )
        h_0 = Variable(torch.zeros(self.bidirectional * self.num_layers, preprocessed_lstm_data.size(0), self.hidden_size)).to(
            self.device
        )

        c_0 = Variable(torch.zeros(self.bidirectional * self.num_layers, preprocessed_lstm_data.size(0), self.hidden_size)).to(
            self.device
        )

        ula, (h_out, _) = self.lstm(preprocessed_lstm_data, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)

        final_input = merge_deep_branch(h_out, indices)
        final_input = merge_wide_branch(final_input, indices)

        linear_out = F.relu(self.linear_4(final_input))
        linear_out = F.relu(self.linear_5(linear_out))
        return linear_out

    
if __name__ == "__main__":
    device = "cpu"
    recurrent_input_size = 73
    hidden_size = 100
    num_layers = 1
    bidirectional=False
    deep_input_size = 16
    # Wide parameters
    wide_input_size = 528

    wdr7 = WDR(
        recurrent_input_size,
        wide_input_size,
        deep_input_size,
        hidden_size,
        num_layers,
        bidirectional,
        device,
    )
    wdr7 = wdr7.to(device)
    wdr7.load_state_dict(torch.load('Abakan/0/Abakan_WDR_model.pth'))
    