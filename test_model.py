import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_preprocess import TrainDataset,test
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from model.gru_model import GRUModel
from model.gru_model2 import GRUModel2
from model.lstm_model import LSTMModel
from model.residuallstm_model import ResLSTM
from model.residualgru_model import ResGRU
from model.model_params import model_params
from train_test_valid import test_fn
import matplotlib.pyplot as plt


def get_model(model,params):
    if model == "GRUModel":
        return GRUModel(params=params)
    elif model == "LSTMModel":
        return LSTMModel(params=params)
    elif model == "GRUModel2":
        return GRUModel2(params=params)
    elif model == "ResLSTM":
        return ResLSTM(input_size=params['input_dim'], hidden_size=[128, 64])
    elif model == "ResGRU":
        return ResGRU(input_size=params['input_dim'], hidden_size=[128, 64, 32])

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Configuration for the training script.')

    # Add arguments
    parser.add_argument('--target_stock', type=str, default='2603', help='Stock for training and testing')
    parser.add_argument('--train_seq_len', type=int, default=5, help='Training sequence length.')
    parser.add_argument('--predict_seq_len', type=int, default=1, help='Prediction sequence length.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training.')
    parser.add_argument('--model', type=str, default='ResGRU', choices=['GRUModel','GRUModel2','LSTMModel','ResLSTM','ResGRU'])

    # Parse arguments
    args = parser.parse_args()
    test_dataset = TrainDataset(test, args.train_seq_len, args.predict_seq_len)
    del test
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    # testing model
    criterion = nn.MSELoss()
    testing_model = get_model(model = args.model,params = model_params)
    testing_model.load_state_dict(torch.load(f'model_state\{args.model}_{args.target_stock}_state_predict_sequence.pth'))
    test_loss, true, pred = test_fn(test_loader, testing_model, criterion, args.device) 
    true = np.concatenate([arr.ravel() for arr in true])
    pred = np.concatenate([arr.ravel() for arr in pred])
    pred_df = pd.DataFrame({'predicted_price':pred,'true_price':true})
    pred_df.to_csv(f'predicted_result\{args.model} {args.target_stock} 2023 predicted prices.csv')
    plot_len = len(true)
    plt.plot(np.arange(plot_len), true, color='blue', label="True Price")
    plt.plot(np.arange(plot_len), pred, color='red', label="Estimated Price")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title("Comparison of Estimated Price and True Price")
    plt.show()

    torch.cuda.empty_cache()
    gc.collect()