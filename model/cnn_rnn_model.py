import torch
from torch import nn
import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from model_params import model_params

class CNN_RNN(nn.Module):
    def __init__(self, nonlinearity="tanh", in_dim=8, in_channels=3, out_dim=1, seq_len=5):
        super(CNN_RNN, self).__init__()
        self.activation = nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=8,
                                           kernel_size=(3, 1),
                                           padding=(1, 0),
                                           dilation=(1, 1)),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=8,
                                           kernel_size=(3, 1),
                                           padding=(2, 0),
                                           dilation=(2, 1)),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn3 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=8,
                                           kernel_size=(3, 1),
                                           padding=(3, 0),
                                           dilation=(3, 1)),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv2d(in_channels=24,
                                           out_channels=16,
                                           kernel_size=(1, 3),
                                           padding=(0, 0)),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv2d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=(1, 3),
                                           padding=(0, 0)),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv2d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=(1, 4)),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation)

        self.linear = nn.Linear(in_features=1, out_features=1)

        self.gru = nn.GRU(input_size=in_channels * in_dim,
                          hidden_size=8,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)

        self.gru_out = nn.GRU(input_size=16,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=16*seq_len, out_features=out_dim)


    def forward(self, x):
        # x:(N, C, T, D)
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1))
        # cnn_out:(N, 16, T, D_in)
        cnn_out = self.linear(cnn_out).squeeze()
        cnn_out = cnn_out.transpose(-1, -2)
        # cnn_out:(N, T, 16)

        x = x.transpose(1, 2)
        # x:(N, T, C, D_in)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        # x:(N, T, C*D_in)
        rnn_out, _ = self.gru(x)
        # rnn_out:(N, T, 16)

        x = rnn_out + cnn_out

        x, _ = self.gru_out(x)
        # x: (N, T, 16)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = self.out(x).squeeze()
        # x:(N, 1)
        return x

# test
if __name__ == '__main__':
    x = torch.rand(1, 3, 5, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_RNN(in_dim=8, in_channels=3, out_dim=1, seq_len=5)
    print(model(x).shape)