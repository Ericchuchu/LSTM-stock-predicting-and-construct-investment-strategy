import torch
from torch import nn
import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from model_params import model_params

class LSTMModel(nn.Module):
    def __init__(self, params):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=params['input_dim'], hidden_size=256, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        self.dense1 = nn.Linear(128, 16)
        self.dropout3 = nn.Dropout(0.3)

        self.dense2 = nn.Linear(16,1)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Only take the output from the final timetep
        x = x[:, -1, :]
        
        x = self.dense1(x)
        x = self.dropout3(x)

        x= self.dense2(x)

        return x
    
if __name__ == '__main__':
    x = torch.rand(1,model_params['timestep'],model_params['input_dim'])
    model = LSTMModel(params=model_params)
    print(model(x).shape)
