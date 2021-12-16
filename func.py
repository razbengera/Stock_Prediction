#this file include function that will be use for the project
import requests
import time
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def stock_data(stock,outputsize="full"):
    """
    :param stock: get string of stock and outputsize = "compact" for 100 days data \ "full" for 20 years data
    :return: csv data file of the stock
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    newpath = r'data'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    api = "7G2MCLFUSWCDEB5J"
    try:
        f = open(os.path.join('data', "{}_daily_{}_{}.csv".format(stock, outputsize, now)), mode ="r")
        f.close()
    except FileNotFoundError:
        r = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}&outputsize={}&datatype=csv".format(stock,api,outputsize))
        if r.text[0] == "t":
            with open(os.path.join('data', "{}_daily_{}_{}.csv".format(stock, outputsize, now)), mode ="wb") as f:
                f.write(r.content)
        else:
            return "ERROR"

def get_forecast(stocks):
    stocks = stocks.split(",")
    for i in stocks:
        stock_data(i)

def last_price(stock,date):
    file_name = str(os.path.join('data', "{}_daily_full_{}.csv".format(stock, date)))
    df = pd.read_csv(file_name)
    df["open"] = df["open"] / (df["close"] / df['adjusted_close'])
    df["low"] = df["low"] / (df["close"] / df['adjusted_close'])
    df["high"] = df["high"] / (df["close"] / df['adjusted_close'])
    df.rename(columns={"timestamp": "time"}, inplace=True)
    df.drop(['close', 'split_coefficient'], axis=1, inplace=True)
    return list(df.iloc[0])

def prep(stock, date,days):
    file_name = str(os.path.join('data', "{}_daily_full_{}.csv".format(stock, date)))
    df = pd.read_csv(file_name)
    df["open"] = df["open"] / (df["close"] / df['adjusted_close'])
    df["low"] = df["low"] / (df["close"] / df['adjusted_close'])
    df["high"] = df["high"] / (df["close"] / df['adjusted_close'])
    df.rename(columns={"timestamp": "time"}, inplace=True)
    df.drop(['close', 'split_coefficient'], axis=1, inplace=True)
    
    df1 = df.copy()
    df1['adjusted_close'][days:] = df1['adjusted_close'][0:len(df1['adjusted_close']) - days]

    drop = list(np.arange(days))
    df1.drop(drop, axis=0, inplace=True)
    return df1.drop(['time','adjusted_close'], axis=1), df1[['adjusted_close']]

def stock_predict(X_train, y_train,data,num):
    sc_x = StandardScaler()
    X_train_s = sc_x.fit_transform(X_train[0:min(num, X_train.shape[0])])
    sc_y = StandardScaler()
    y_train_s = sc_y.fit_transform(y_train[0:min(num, y_train.shape[0])])

    model = LinearRegression()
    model.fit(X_train_s, y_train_s)
    normal_value = sc_x.transform([data])
    res = model.predict(normal_value)
    return "{:.2f}".format(float(sc_y.inverse_transform(res)[0]))

def stock_forest_predict(X_train, y_train,data):
    sc_x = MinMaxScaler()
    X_train_s = sc_x.fit_transform(X_train)
    sc_y = MinMaxScaler()
    y_train_s = sc_y.fit_transform(y_train)

    rfc = RandomForestRegressor(n_estimators=1000, min_samples_leaf=8)
    rfc.fit(X_train_s, y_train_s)
    normal_value = sc_x.transform([data])
    res = rfc.predict(normal_value)
    return "{:.2f}".format(float(sc_y.inverse_transform([res])[0]))

def NN_predict(X_train, y_train,data):
    X_train["bias"] = 1
    mlp = MLPRegressor(activation="relu", learning_rate_init=0.00001, alpha=1e-5,
                       hidden_layer_sizes=(40, 30, 20), max_iter=100000, random_state=111)
    mlp.fit(X_train, y_train)
    return "{:.2f}".format(float(mlp.predict([data])[0]))

def stock_graph(stock, date):
    file_name = str(os.path.join('data', "{}_daily_full_{}.csv".format(stock, date)))
    df = pd.read_csv(file_name)
    x = df["timestamp"][::-1]
    y = df["adjusted_close"][::-1]
    plt.plot(x, y)
    plt.savefig(str(os.path.join('static', '{}.jpg'.format(stock.upper()))))

def stock_data_for_graph(stock, date, days):
    file_name = str(os.path.join('data', "{}_daily_full_{}.csv".format(stock, date)))
    df = pd.read_csv(file_name)
    data_str = ""
    for i in range(days):
        data_str+="{ x: new Date("+str(df["timestamp"][i])[0:4]+", "+str(int(str(df["timestamp"][i])[5:7])-1)+", "+str(df["timestamp"][i])[8:10]+"), y: "+str(df["adjusted_close"][i])+" },"
    data_str = data_str[:-1]
    return data_str

if __name__ == '__main__':
    stocks = ["ibm","aapl","rfdf","uso","tqqq"]
    for i in stocks:
        stock_data(i)


