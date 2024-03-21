import twstock
import requests
import pandas as pd

def get_data(target_stock):
    # finmind 抓取三大法人買賣資料
    url = "https://api.finmindtrade.com/api/v3/data"
    parameter = {
        "dataset": "InstitutionalInvestorsBuySell",
        "stock_id": {target_stock},
        "date": "2016-01-01",
        "end_date": "2022-12-31",
    }
    data_investor = requests.get(url, params=parameter)
    data_investor = data_investor.json()
    data_investor = pd.DataFrame(data_investor['data'])
    # 轉換為寬數據
    data_investor = data_investor.pivot_table(index=["date", "stock_id"], columns="name", values=["buy", "sell"], fill_value=0).reset_index()
    # 調整列名
    data_investor.columns = ['_'.join(col).strip() for col in data_investor.columns.values]
    data_investor.rename(columns ={"date_":"Date","stock_id_":"stock_id"},inplace=True)
    data_investor['Date'] = pd.to_datetime(data_investor['Date'])
    data_investor.index = data_investor['Date']
    data_investor.drop(columns=["stock_id","buy_Foreign_Dealer_Self","sell_Foreign_Dealer_Self","Date"], inplace=True)

    # twstock 抓取股價
    stock = twstock.Stock(target_stock)  
    name_attribute = [
        'Date', 'Capacity', 'Turnover', 'Open', 'High', 'Low', 'Close', 'Change',
        'Transcation'
    ]  
    # 取時間period的資料
    data_price = pd.DataFrame(columns=name_attribute)
    for year in range(2016, 2023): 
        for month in range(1, 13): 
            monthly_data = stock.fetch(year, month)
            df_month = pd.DataFrame(columns=name_attribute, data=monthly_data)
            data_price = pd.concat([data_price, df_month], ignore_index=True)
    data_price['Date'] = pd.to_datetime(data_price['Date'])
    data_price.index = data_price['Date']
    data_price.drop(columns=["Date"], inplace=True)

    print(len(data_investor))
    print(len(data_price))
    # 合併兩個 dataframe
    merged_df = pd.merge(data_price, data_investor, on='Date', how='inner')
    filename = f'data\{target_stock} 2016-2022.csv'
    merged_df.to_csv(filename)
    #將Data Frame轉存為csv檔案

    # finmind 抓取三大法人買賣資料
    url = "https://api.finmindtrade.com/api/v3/data"
    parameter = {
        "dataset": "InstitutionalInvestorsBuySell",
        "stock_id": {target_stock},
        "date": "2023-01-01",
        "end_date": "2023-12-31",
    }
    data_investor = requests.get(url, params=parameter)
    data_investor = data_investor.json()
    data_investor = pd.DataFrame(data_investor['data'])
    # 轉換為寬數據
    data_investor = data_investor.pivot_table(index=["date", "stock_id"], columns="name", values=["buy", "sell"], fill_value=0).reset_index()
    # 調整列名
    data_investor.columns = ['_'.join(col).strip() for col in data_investor.columns.values]
    data_investor.rename(columns ={"date_":"Date","stock_id_":"stock_id"},inplace=True)
    data_investor['Date'] = pd.to_datetime(data_investor['Date'])
    data_investor.index = data_investor['Date']
    data_investor.drop(columns=["stock_id","buy_Foreign_Dealer_Self","sell_Foreign_Dealer_Self","Date"], inplace=True)

    # twstock 抓取股價
    stock = twstock.Stock(target_stock)  
    name_attribute = [
        'Date', 'Capacity', 'Turnover', 'Open', 'High', 'Low', 'Close', 'Change',
        'Transcation'
    ]  
    # 取時間period的資料
    data_price = pd.DataFrame(columns=name_attribute)
    for year in range(2023, 2024): 
        for month in range(1, 13): 
            monthly_data = stock.fetch(year, month)
            df_month = pd.DataFrame(columns=name_attribute, data=monthly_data)
            data_price = pd.concat([data_price, df_month], ignore_index=True)
    data_price['Date'] = pd.to_datetime(data_price['Date'])
    data_price.index = data_price['Date']
    data_price.drop(columns=["Date"], inplace=True)

    print(len(data_investor))
    print(len(data_price))
    # 合併兩個 dataframe
    merged_df = pd.merge(data_price, data_investor, on='Date', how='inner')
    filename = f'data\{target_stock} 2023.csv'
    merged_df.to_csv(filename)
    #將Data Frame轉存為csv檔案

if __name__ == '__main__':
    data = pd.read_csv(r'2023籌碼選股\selected_stock.csv')
    selected_stock = data['股票代號'].to_numpy()
    for stock in selected_stock:
        get_data(str(stock))
