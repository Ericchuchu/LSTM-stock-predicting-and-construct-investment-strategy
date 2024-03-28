import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_preprocess import TrainDataset,train,valid,test
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
from train_test_valid import train_fn,valid_fn,test_fn
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
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=4e-3, help='Learning rate for optimizer.')
    parser.add_argument('--model', type=str, default='ResGRU', choices=['GRUModel','GRUModel2','LSTMModel','ResLSTM','ResGRU'])

    # Parse arguments
    args = parser.parse_args()
    train_dataset = TrainDataset(train, args.train_seq_len, args.predict_seq_len)
    del train
    valid_dataset = TrainDataset(valid, args.train_seq_len, args.predict_seq_len)
    del valid
    test_dataset = TrainDataset(test, args.train_seq_len, args.predict_seq_len)
    del test

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = get_model(model = args.model, params=model_params)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay= 0.005)
    criterion = nn.MSELoss()
    best_loss_mse = float('inf')
    train_loss_curve = []
    valid_loss_curve = []

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_fn(train_loader, model, criterion, optimizer, epoch, device=args.device)
        valid_loss = valid_fn(valid_loader, model, criterion, args.device)  
        elapsed = time.time() - start_time

        print(
            f" Epoch {epoch+1} - avg time: {elapsed:.0f}s \n"
            f" avg_train_MSEloss: {train_loss.avg_mse:.4f} - avg_train_MAPEloss: {train_loss.avg_mape:.4f}\n"
            f" avg_val_MSEloss: {valid_loss.avg_mse:.4f} - avg_val_MAPEloss: {valid_loss.avg_mape:.4f}\n" 
        )

        train_loss_mse = train_loss.avg_mse
        valid_loss_mse = valid_loss.avg_mse 
        train_loss_curve.append(train_loss_mse)
        valid_loss_curve.append(valid_loss_mse)

        # 如果需要保存模型
        if valid_loss_mse < best_loss_mse:
            torch.save(model.state_dict(),f'model_state\{args.model}_{args.target_stock}_state_predict_sequence.pth')
            best_loss_mse = valid_loss_mse

    # visualize loss curve
    loss_curve_len = len(train_loss_curve)
    plt.plot(range(1,(loss_curve_len)+1), train_loss_curve, color='blue', label="train_mse")
    plt.plot(range(1,(loss_curve_len)+1), valid_loss_curve, color='red', label="valid_mse")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Comparison of train loss and valid loss curve")
    plt.show()
    
    torch.cuda.empty_cache()
    gc.collect()