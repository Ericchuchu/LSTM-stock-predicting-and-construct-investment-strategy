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

    return data

class TrainDataset(Dataset):
    def __init__(self, data, train_seq_len, predict_seq_len):  
        self.data_idxs = None 
        self.data = data    
        self.train_seq_len = train_seq_len
        self.predict_seq_len = predict_seq_len
        # 'Capacity', 'Turnover', 'Open', 'High', 'Low', 'Close', 'Change','Transcation', 'buy_Dealer_Hedging', 'buy_Dealer_self','buy_Foreign_Investor', 'buy_Investment_Trust', 'sell_Dealer_Hedging','sell_Dealer_self', 'sell_Foreign_Investor', 'sell_Investment_Trust'
        # 'Turnover','Open','High','Low','Close','SAR','upper_channel','lower_channel','upper_bb','lower_bb','moving_average_5','moving_average_10','moving_average_20','squeeze','momentum'
        self.feature_cols = ['Capacity','Open','High','Low','Close','Change','upper_channel','lower_channel','upper_bb',
                             'lower_bb','moving_average_5','momentum','buy_Dealer_Hedging', 'buy_Dealer_self','buy_Foreign_Investor', 
                             'buy_Investment_Trust','sell_Dealer_Hedging','sell_Dealer_self', 'sell_Foreign_Investor','sell_Investment_Trust']
        # 計算有效的索引範圍
        self.data_idxs = list(range(len(data) - train_seq_len - predict_seq_len))

    def __len__(self):
        return len(self.data_idxs)
    
    def __getitem__(self, idx):
        # 從有效索引範圍中選擇開始點
        start = self.data_idxs[idx]
        end = start + self.train_seq_len
        
        # 輸入的特徵以及標籤
        sample = self.data.iloc[start:end]
        sample.loc[:, self.feature_cols] = sample[self.feature_cols].fillna(sample[self.feature_cols].median()).fillna(0)
        inputs = torch.tensor(sample[self.feature_cols].values, dtype=torch.float)
        
        label_start = end
        label_end = end + self.predict_seq_len
        labels = self.data.iloc[label_start:label_end]['Close']
        labels = labels.fillna(labels.median()).fillna(0).values
        labels = torch.tensor(labels, dtype=torch.float)
        
        return inputs, labels

# training 跟 testing 資料
data_train_valid = data_preprocess(r"data\3105 2016-2022.csv",'train')
test = data_preprocess(r"data\3105 2023.csv", 'test')
# Use random_split to split the dataset into two parts
train = data_train_valid.iloc[:int(len(data_train_valid)*0.8),:] #1355
valid = data_train_valid.iloc[int(len(data_train_valid)*0.8):,:] #339
print(len(train)) 
print(len(valid))
del data_train_valid
# train_dataset = TrainDataset(train, CFG.train_seq_len, CFG.predict_seq_len)
test_dataset = TrainDataset(test,CFG.train_seq_len, CFG.predict_seq_len)
# print(train_dataset[1])
gc.collect()
