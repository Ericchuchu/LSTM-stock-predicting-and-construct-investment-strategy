import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from data_preprocess_3channels import train_input, train_label, val_input, val_label
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from model.convlstm_model import ConvLSTM
from train_test_valid import train_fn,valid_fn
import matplotlib.pyplot as plt

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Configuration for the training script.')

    # Add arguments
    parser.add_argument('--target_stock', type=str, default='3037', help='Stock for training and testing')
    parser.add_argument('--train_seq_len', type=int, default=5, help='Training sequence length.')
    parser.add_argument('--predict_seq_len', type=int, default=1, help='Prediction sequence length.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer.')

    # Parse arguments
    args = parser.parse_args()

    # 合併為一個channel
    # train_input = train_input.reshape(train_input.shape[0], 1, train_input.shape[2], train_input.shape[1] * train_input.shape[3])
    # val_input = val_input.reshape(val_input.shape[0], 1, val_input.shape[2], val_input.shape[1] * val_input.shape[3])

    train_loader = DataLoader(TensorDataset(train_input, train_label), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(val_input, val_label), batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = ConvLSTM(input_channels=3, hidden_channels=[8, 1], kernel_size=5, in_dim=8, out_dim=1, device=args.device, step=5)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay= 0.005)
    criterion = nn.MSELoss()
    best_loss_mse = float('inf')
    train_loss_curve = []
    valid_loss_curve = []

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_fn(train_loader, model, criterion, optimizer, epoch, device=args.device)
        valid_loss = valid_fn(val_loader, model, criterion, args.device)  
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
            torch.save(model.state_dict(),f'model_state\Conv-LSTM_{args.target_stock}_state_predict_sequence.pth')
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