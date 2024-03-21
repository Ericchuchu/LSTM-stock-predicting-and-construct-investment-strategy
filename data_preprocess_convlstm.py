import os
import gc
import numpy as np
import pandas as pd
import random
import torch
import pandas as pd
import talib
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# setup configuration
# 用五天的資料去預測下一天的收盤價
class CFG:
    train_seq_len = 5
    predict_seq_len = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_preprocess(data_path, type):
    # 計算keltner channel
    def keltner_channel(high, low, close, timeperiod = 20, nbdev = 2):
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
    upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=19, nbdevup=2.44, nbdevdn=2.44, matype=0)
    sar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    momentum = talib.MOM(data['Close'], timeperiod=10)
    moving_average_5 = talib.SMA(data['Close'], timeperiod=5)
    moving_average_10 = talib.SMA(data['Close'], timeperiod=10)
    moving_average_20 = talib.SMA(data['Close'], timeperiod=20)

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

    # 進行標準化，除了squeeze
    data.drop(columns = ['squeeze'], inplace = True)
    # 所有 data 進行標準化
    
    if type == 'train':
        scaler = MinMaxScaler(feature_range=(0,1))
        data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns, index = data.index)
        data['squeeze'] = data_squeeze.astype(int)
        # 保存scaler
        dump(scaler, 'scaler.save')

    if type == 'test':
        scaler = load('scaler.save')
        data = pd.DataFrame(scaler.transform(data), columns = data.columns, index = data.index)
        data['squeeze'] = data_squeeze.astype(int)

    feature_cols = ['Capacity','Turnover','Transcation','Open','High','Low','Close','Change',
                    'upper_channel','lower_channel','upper_bb','lower_bb','moving_average_5','moving_average_10','moving_average_20','momentum', 
                    'buy_Dealer_Hedging','buy_Dealer_self','buy_Foreign_Investor', 'buy_Investment_Trust','sell_Dealer_Hedging','sell_Dealer_self', 'sell_Foreign_Investor','sell_Investment_Trust']
    data = data[feature_cols]
    data.reset_index(drop=True, inplace = True)
    # reshape
    first_channel = ['Capacity','Turnover','Transcation','Open','High','Low','Close','Change']
    second_channel = ['upper_channel','lower_channel','upper_bb','lower_bb','moving_average_5','moving_average_10','moving_average_20','momentum']
    third_channel = ['buy_Dealer_Hedging','buy_Dealer_self','buy_Foreign_Investor', 'buy_Investment_Trust','sell_Dealer_Hedging','sell_Dealer_self', 'sell_Foreign_Investor','sell_Investment_Trust']
    seq_len = 5
    if type == 'train':
        train_input = []
        train_label = []
        for index in range(data.shape[0] - seq_len):
            input = torch.zeros((3, 5, 8))
            # 分為三個 channel 
            tmp_input = data.iloc[index:index+seq_len+1]
            tmp_input.reset_index(drop=True, inplace=True)
            input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][first_channel], dtype=np.float32))
            input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][second_channel], dtype=np.float32))
            input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][third_channel], dtype=np.float32))
            label = torch.tensor(np.array(tmp_input.loc[seq_len]['Close'], dtype=np.float32))
            train_input.append(input)
            train_label.append(label)

        # 將 train_input 轉為 torch tensor
        zeros = torch.zeros((len(train_input), 3, 5, 8))
        for i in range(zeros.shape[0]):
            zeros[i] = train_input[i]
        train_input = zeros
        # print("Shape of the Training Input:", train_input.shape)

        train_label = torch.tensor(train_label)
        # print("Shape of the Training Label:", train_label.shape)

        return train_input,train_label

    if type == 'test':     
        test_input = []
        test_label = []

        for index in range(data.shape[0] - seq_len):
            input = torch.zeros((3, 5, 8))
            # 分為三個 channel 
            tmp_input = data.iloc[index:index+seq_len+1]
            tmp_input.reset_index(drop=True, inplace=True)
            input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][first_channel], dtype=np.float32))
            input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][second_channel], dtype=np.float32))
            input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][third_channel], dtype=np.float32))
            label = torch.tensor(np.array(tmp_input.loc[seq_len]['Close'], dtype=np.float32))
            test_input.append(input)
            test_label.append(label)

        # 將 test_input 轉為 torch tensor
        zeros = torch.zeros((len(test_input), 3, 5, 8))
        for i in range(zeros.shape[0]):
            zeros[i] = test_input[i]
        test_input = zeros
        # print("Shape of the Testing Input:", test_input.shape)

        test_label = torch.tensor(test_label)
        # print("Shape of the Testing Label:", test_label.shape)

        return test_input,test_label

# training 跟 testing 資料
train_input, train_label = data_preprocess(r"data\00637L 2016-2022.csv",'train')
val_input, val_label = train_input[int(len(train_input)*0.8):],train_label[int(len(train_label)*0.8):]
train_input, train_label = train_input[:int(len(train_input)*0.8)],train_label[:int(len(train_label)*0.8)]
test_input, test_label = data_preprocess(r"data\00637L 2023.csv", 'test')
# Shape of the Training Input: torch.Size([1680, 3, 5, 8])
# Shape of the Training Label: torch.Size([1680])
# Shape of the Testing Input: torch.Size([214, 3, 5, 8])
# Shape of the Testing Label: torch.Size([214])

