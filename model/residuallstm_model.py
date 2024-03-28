import torch
import torch.nn as nn
from torch.autograd import Variable


class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, residual_on = True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ResLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual_on = residual_on
        self.weight_i = nn.Parameter(torch.rand(3*hidden_size, input_size)).to(device)
        self.weight_h = nn.Parameter(torch.rand(3*hidden_size, hidden_size)).to(device)
        self.weight_c = nn.Parameter(torch.rand(3*hidden_size, hidden_size)).to(device)
        self.bias_i = nn.Parameter(torch.rand(3*hidden_size)).to(device)
        self.bias_h = nn.Parameter(torch.rand(3*hidden_size)).to(device)
        self.bias_c = nn.Parameter(torch.rand(3*hidden_size)).to(device)
        self.weight_cch = nn.Parameter(torch.rand(1*hidden_size, hidden_size)).to(device)
        self.bias_cch = nn.Parameter(torch.rand(1*hidden_size)).to(device)
        self.weight_ccx = nn.Parameter(torch.rand(1*hidden_size, input_size)).to(device)
        self.bias_ccx = nn.Parameter(torch.rand(1*hidden_size)).to(device)
        self.weight_ir = nn.Parameter(torch.rand(hidden_size, input_size)).to(device)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        info_gate = (torch.mm(x, self.weight_i.t()) + self.bias_i +
                     torch.mm(h, self.weight_h.t()) + self.bias_h +
                     torch.mm(c, self.weight_c.t()) + self.bias_c )
        
        ingate, forgetgate, outgate = info_gate.chunk(3,1)
        cellgate = (torch.mm(h, self.weight_cch.t()) + self.bias_cch + torch.mm(x, self.weight_ccx.t()) + self.bias_ccx)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cc = (forgetgate * c) + (ingate * cellgate)
        ry = torch.tanh(cc)
        
        if self.residual_on:
            if self.input_size == self.hidden_size:
                ch = outgate*ry + x
            else:
                ch = outgate*(ry + torch.mm(x, self.weight_ir.t()))
            return ch, cc
        else:
            ch = outgate*ry
            return ch,cc

    def init_hidden(self, batch_size, dim, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        return (Variable(torch.zeros(batch_size, dim)).to(device),
                Variable(torch.zeros(batch_size, dim)).to(device))


class ResLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_size, hidden_size, attention = False, out_dim = 1, step = 5, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ResLSTM, self).__init__()
        self.input_size = [input_size] + hidden_size
        self.hidden_size = hidden_size
        self.num_layers = len(hidden_size)
        self.step = step
        self.device = device
        self.attention = attention
        self._all_layers = []
        self.dropout_layers = nn.ModuleList([nn.Dropout(0.01) for hs in hidden_size]) 
        # 加入attention注意力機制
        self.attention_layer = nn.Linear(hidden_size[-1], hidden_size[-1])
        self.linear = nn.Linear(in_features = hidden_size[-1], out_features = out_dim)

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i == 0:
                cell = ResLSTMCell(self.input_size[i], self.hidden_size[i])
            if i > 0:
                cell = ResLSTMCell(self.input_size[i], self.hidden_size[i], residual_on=False)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        # input:(N, T, D_in)
        internal_state = []
        for step in range(self.step):
            x = input[:, step, :]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize,_ = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize,dim=self.hidden_size[i], device = self.device)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                x = self.dropout_layers[i](x)
                internal_state[i] = (x, new_c)

            outputs = x
            # outputs:(N, D_in)

            if step == (self.step - 1) and self.attention:
                # 注意力層
                attention_weights = torch.softmax(self.attention_layer(outputs), dim=1)
                attention_output = torch.sum(attention_weights * outputs, dim=1)
                return attention_output.unsqueeze(-1)
            
            else :
                outputs = self.linear(outputs)
                # outputs:(N, D_out)
           
        return outputs

# test
if __name__ == '__main__':
    x = torch.rand(1, 5, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResLSTM(input_size=8, hidden_size=[256, 128, 64], out_dim=1, step = 5, device = device)
    print(model(x).shape)