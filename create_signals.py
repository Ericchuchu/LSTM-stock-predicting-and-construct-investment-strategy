import os
import gc
import numpy as np
import pandas as pd
import argparse
import torch
import pandas as pd
import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model.gru_model import GRUModel
from model.gru_model2 import GRUModel2
from model.lstm_model import LSTMModel
from model.residualgru_model import ResGRU
from model.residuallstm_model import ResLSTM
from model.model_params import model_params
from data_preprocess import data_preprocess
from joblib import load

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

def data_preprocess_signal(data_path, seq):
    # 計算keltner channel
    def keltner_channel(high, low, close, timeperiod = 20, nbdev = 1.5):
        middle_line = talib.EMA(close, timeperiod = timeperiod)
        atr = talib.ATR(high, low, close, timeperiod = timeperiod)
        
        # 上通道和下通道
        upper_channel = middle_line + nbdev * atr
        lower_channel = middle_line - nbdev * atr
        
        return upper_channel, middle_line, lower_channel

    # prepare data
    data = pd.read_csv(data_path, encoding = 'utf-8', encoding_errors = 'ignore')
    data = data.set_index('Date')
    data.index = pd.to_datetime(data.index)

    # 計算指標
    upper_channel, middle_line, lower_channel = keltner_channel(data['High'], data['Low'], data['Close'])
    upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=1)
    sar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    momentum = talib.MOM(data['Close'], timeperiod=10)
    moving_average_5 = talib.EMA(data['Close'], timeperiod=5)
    moving_average_10 = talib.EMA(data['Close'], timeperiod=10)
    moving_average_20 = talib.EMA(data['Close'], timeperiod=20)

    # 放入dataframe
    data['SAR'] = sar
    data['upper_channel'] = upper_channel
    data['lower_channel'] = lower_channel
    data['upper_bb'] = upper
    data['lower_bb'] = lower
    data['moving_average_5'] = moving_average_5
    data['moving_average_10'] = moving_average_10
    data['moving_average_20'] = moving_average_20
    data['momentum'] = momentum
    data['squeeze'] = (data['upper_bb'] <= data['upper_channel']) & (data['lower_bb']  >= data['lower_channel'])
    data_squeeze = data['squeeze']


    # 把前20筆資料drop掉
    data = data[20:]
    
    # 前 sequence 資料
    data = data[-seq:]

    # 進行標準化，除了squeeze
    data.drop(columns = ['squeeze'], inplace = True)
    # 所有 data 進行標準化
    # 對 train_data 進行 min max scaler 並儲存
    _ = data_preprocess(data_path, 'train')
    scaler = load('scaler.save')
    data = pd.DataFrame(scaler.transform(data), columns = data.columns, index = data.index)
    data['squeeze'] = data_squeeze.astype(int)

    return data

def data_preprocess_backtesting(stock_number, new_true_price, new_predicted_price, threshold = 0):
    stock_df = pd.read_csv(f'C:/Users/111030/Desktop/strategy_online/LSTM_stock_predicting_strategy/LSTM-stock-predicting-and-construct-investment-strategy-main/data/{stock_number} 2024-now.csv', encoding='utf-8', encoding_errors= 'ignore')
    five_day_min_low = talib.MIN(stock_df['Low'], timeperiod=5) # 五天最低價當作long strategy的停損點
    five_day_max_high = talib.MAX(stock_df['High'], timeperiod=5) # 五天最高價當作short strategy的停損點
    stock_df['ADX'] = talib.ADX(stock_df['High'], stock_df['Low'], stock_df['Close'], timeperiod=14)
    stock_df['five_day_min_low'] = five_day_min_low
    stock_df['five_day_max_high'] = five_day_max_high
    stock_df = stock_df[25:]
    predicted_price_df = pd.read_csv(f'C:/Users/111030/Desktop/strategy_online/LSTM_stock_predicting_strategy/LSTM-stock-predicting-and-construct-investment-strategy-main/predicted_result/ResGRU {stock_number} 2024-now predicted prices.csv', encoding='utf-8', encoding_errors = 'ignore')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.index = stock_df['Date']
    new_index = len(predicted_price_df)
    predicted_price_df.loc[new_index] = {
        'predicted_price': new_predicted_price,
        'true_price': new_true_price,
    }
    stock_df['predicted_price'] = predicted_price_df['predicted_price'].values
    stock_df['true_price'] = predicted_price_df['true_price'].values

    # long short condition
    stock_df['predicted_price_up'] = (stock_df['predicted_price'] > stock_df['predicted_price'].shift(1)).astype(int)
    stock_df['actual_price_up'] = (stock_df['Close'] > stock_df['Close'].shift(1)).astype(int)
    stock_df['long_condition'] = (stock_df['true_price'].shift(1)<=stock_df['predicted_price']) & ((stock_df['predicted_price']-stock_df['true_price'].shift(1))/stock_df['true_price'].shift(1) >= threshold) 
    stock_df['short_condition'] = (stock_df['true_price'].shift(1)>stock_df['predicted_price']) & (abs((stock_df['predicted_price']-stock_df['true_price'].shift(1))/stock_df['true_price'].shift(1)) >= threshold)
    stock_df['long_condition'] = stock_df['long_condition'].astype(int)
    stock_df['short_condition'] = stock_df['short_condition'].astype(int)
    
    return stock_df

# create signals
def create_signals(stock_df):
    signals_long = pd.Series(np.zeros(len(stock_df)), index=stock_df.index)
    partition_size_long = 0
    stop_loss_long = 0
    for i in range(len(stock_df)):
        if partition_size_long == 0:
            if stock_df['long_condition'][i] == 1:
                signals_long[i] = 1
                partition_size_long = 1
                stop_loss_long = stock_df['five_day_min_low'][i]
        if partition_size_long == 1:
            if (stock_df['Close'][i] <= stop_loss_long) or (stock_df['short_condition'][i] == 1): # 停利停損
                signals_long[i] = -1
                stop_loss_long = 0 # 停損點重置
                partition_size_long = 0 # 平倉

    # short strategy
    signals_short = pd.Series(np.zeros(len(stock_df)), index=stock_df.index)
    partition_size_short = 0
    stop_loss_short = 0
    for i in range(len(stock_df)):
        if partition_size_short == 0:
            if stock_df['short_condition'][i] == 1:
                signals_short[i] = -1
                partition_size_short = -1
                stop_loss_short = stock_df['five_day_max_high'][i]
        if partition_size_short == -1:
            if (stock_df['Close'][i] >= stop_loss_short) or (stock_df['long_condition'][i] == 1):# 停利停損
                signals_short[i] = 1
                stop_loss_short = 0 # 停損點重置
                partition_size_short = 0 # 平倉

    entries_long = signals_long == 1
    exits_long = signals_long == -1
    entries_short = signals_short == -1
    exits_short = signals_short == 1
    
    return entries_long, exits_long, entries_short, exits_short, stock_df.index[-1]


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Configuration for the training script.')

    # Add arguments
    parser.add_argument('--seq_len', type=int, default=5, help='Training sequence length.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training.')
    parser.add_argument('--model', type=str, default='ResGRU', choices=['GRUModel','GRUModel2','LSTMModel','ResLSTM','ResGRU'])

    # Parse arguments
    args = parser.parse_args()
    
    data = pd.read_csv(r'C:\Users\111030\Desktop\strategy_online\LSTM_stock_predicting_strategy\select_stock_2024\selected_stock\selected_stock.csv')
    selected_stock = data['stock_number'].to_numpy()
    
    feature_cols = ['Capacity','Open','High','Low','Close','Change','upper_channel','lower_channel','upper_bb',
                'lower_bb','moving_average_5','momentum','buy_Dealer_Hedging', 'buy_Dealer_self','buy_Foreign_Investor', 
                'buy_Investment_Trust','sell_Dealer_Hedging','sell_Dealer_self', 'sell_Foreign_Investor','sell_Investment_Trust']
    signals_dict = {'stock':[],'entries_long':[],'exits_long':[],'entries_short':[],'exits_short':[]}
    
    for stock in selected_stock:
        print(f'{stock} testing the model')
        data = data_preprocess_signal(f"data/{stock} 2024-now.csv",args.seq_len)
        inputs = torch.tensor(data[feature_cols].values, dtype=torch.float)
        inputs = inputs.unsqueeze(0)
        model = get_model(model = args.model,params = model_params)
        model.load_state_dict(torch.load(f'model_state\{args.model}_{stock}_state_predict_sequence.pth'))
        model.eval()
        inputs = inputs.to(args.device)    
        with torch.no_grad():
            prediction = model(inputs)        
        prediction = float(np.concatenate(prediction.detach().to('cpu').numpy()))
        last_close_price = data['Close'].iloc[-1]
        stock_df = data_preprocess_backtesting(stock, last_close_price, prediction)
        entries_long, exits_long, entries_short, exits_short, timestamp= create_signals(stock_df)
        signals_dict['stock'].append(stock)
        signals_dict['entries_long'].append(entries_long.iloc[-1])
        signals_dict['exits_long'].append(exits_long.iloc[-1])
        signals_dict['entries_short'].append(entries_short.iloc[-1])
        signals_dict['exits_short'].append(exits_short.iloc[-1])

    signals_df = pd.DataFrame(signals_dict)
    print(timestamp)
    print(signals_df)
