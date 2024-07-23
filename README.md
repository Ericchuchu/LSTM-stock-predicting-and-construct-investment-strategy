# LSTM-stock-predicting-and-construct-investment-strategy
This project employs four different deep learning architectures (LSTM, Conv-LSTM, GRU, GRU2) to predict stock prices and develop investment trading strategies. Currently, it utilizes Taiwan stocks for backtesting. The selection of individual stocks is primarily based on the proportion of institutional holdings and trading volume, although this criterion may be refined and modified in the future. At present, these models are still in their initial stages, but the trading performance of individual stocks generally shows an annualized return rate of 50-100%, with a Sharpe ratio consistently above 2.5. Detailed backtesting results can be viewed in the HTML file within the trade folder. Finally, the project aims to optimize trading decisions by identifying parameter plateaus through multiple trials.

## Data
Use APIs to fetch stock prices, institutional net buying/selling data, etc., and independently calculate technical indicators to serve as features for model training. The final set of features determined includes 'Capacity', 'Open', 'High', 'Low', 'Close', 'Change', 'upper_channel', 'lower_channel', 'upper_bb', 'lower_bb', 'moving_average_5', 'momentum', 'buy_Dealer_Hedging', 'buy_Dealer_self', 'buy_Foreign_Investor', 'buy_Investment_Trust', 'sell_Dealer_Hedging', 'sell_Dealer_self', 'sell_Foreign_Investor', 'sell_Investment_Trust

## Model
1. LSTM : a 2-layers LSTM model forecasting stock prie from the time series data

2. GRU : a 2-layers and a 3-layers GRU model forecasting stock prie from the time series data

3. Conv-LSTM : LSTM constructed with convolutional 1D neural network in order to get more informations from the relationship between different feature

4. CNN + RNN : a 3-layer GRU model combine with convolutional 1D neural networks(with different dilations), and their blocks are independent to others. The outputs from different blocks combined and put into the final block contained with 1-layer GRU and linear regression layer

5. Residual GRU : 3-layers GRU model embeded with residual network block

6. Residual LSTM : 2-layers LSTM model embeded with residual network block

## Train and validate
Cross validation : has been applied cross validation method on the training progress, but the performance grtting worse, so eventually I did not use cross validation method, the cause of the effect may comes to the time series data, which does not match thr concept of cross validation 

## Trade
Long condition : When predicted stock close price higher than the close price of previous day (setting a threshold to filter better signals), using open price to buy the stock tomorrow and setting the minimum price in five days before the signals as stop loss point

Short condition : When predicted stock close price lower than the close price of previous day (setting a threshold to filter better signals), using open price to sell the stock tomorrow and setting the maxmium price in five days before the signals as stop loss point

## Result
The backtesting result of the individual stocks were stored in .html in the trade file

## article
https://medium.com/@eric07310115/deep-learning-driven-cta-investment-decisions-stock-price-prediction-and-portfolio-management-9b676da6d2f7
