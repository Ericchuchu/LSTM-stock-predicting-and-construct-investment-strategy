import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_preprocess_3channels import test_input, test_label
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from model.cnn_rnn_model import CNN_RNN
from model.model_params import model_params
from train_test_valid import test_fn
import matplotlib.pyplot as plt


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Configuration for the training script.')

    # Add arguments
    parser.add_argument('--target_stock', type=str, default='2603', help='Stock for training and testing')
    parser.add_argument('--train_seq_len', type=int, default=5, help='Training sequence length.')
    parser.add_argument('--predict_seq_len', type=int, default=1, help='Prediction sequence length.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training.')

    # Parse arguments
    args = parser.parse_args()

    test_loader = DataLoader(TensorDataset(test_input, test_label), batch_size=args.batch_size, shuffle=False, drop_last=False)
    # testing model
    criterion = nn.MSELoss()
    testing_model = CNN_RNN(in_dim=8, in_channels=3, out_dim=1, seq_len=5)
    testing_model.to(args.device)
    state_dict = torch.load(f'model_state\CNN_RNN_{args.target_stock}_state_predict_sequence.pth')
    testing_model.load_state_dict(state_dict, strict=False)
    test_loss, true, pred = test_fn(test_loader, testing_model, criterion, args.device) 
    true = np.concatenate([arr.ravel() for arr in true])
    pred = np.concatenate([arr.ravel() for arr in pred])
    pred_df = pd.DataFrame({'predicted_price':pred,'true_price':true})
    pred_df.to_csv(f'predicted_result\CNN_RNN {args.target_stock} 2023 predicted prices.csv')
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