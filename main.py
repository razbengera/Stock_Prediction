from flask import Flask, render_template, request
import datetime
import os
import func

app = Flask(__name__)
#"http://localhost:8080/"
@app.route('/')
def index():
    return render_template('form1.html')

@app.route('/getstock')
def get_stock():
    stock = request.args.get('stock')
    #download data
    stock_data = func.stock_data(stock)
    #if stock doesn't exist
    if stock_data == "ERROR":
        return render_template('error.html', stock=stock.upper())
    else:
        pass

    #last price
    last_price = func.last_price(stock, datetime.datetime.now().strftime("%Y-%m-%d"))
    [date, last_open, last_high, last_low, last_adjusted_close, last_volume, dividend_amount] = last_price

    return render_template('form2.html',stock = stock, date = date, last_open = last_open,
                           last_high = last_high, last_low = last_low,last_adjusted_close = last_adjusted_close,
                           last_volume = last_volume, dividend_amount = dividend_amount)

@app.route('/getForecast')
def get_forecast():
    stock = request.args.get('stock')
    days = int(request.args.get('days'))
    open_price = float(request.args.get('open_price'))
    high_price = float(request.args.get('high_price'))
    low_price = float(request.args.get('low_price'))
    adjusted_close = func.last_price(stock, datetime.datetime.now().strftime("%Y-%m-%d"))[4]
    volume = float(request.args.get('volume'))
    dividend_amount = float(request.args.get('dividend_amount'))



    data = [open_price,high_price,low_price,volume,dividend_amount]


    #preper the data
    X_train, y_train = func.prep(stock, datetime.datetime.now().strftime("%Y-%m-%d"), days)
    #predict
    pred = float(func.stock_forest_predict(X_train, y_train, data))*0.6 + float(func.stock_predict(X_train, y_train,data,60))*0.15 + float(func.stock_predict(X_train, y_train,data,30))*0.25
    data_graph = func.stock_data_for_graph(stock, datetime.datetime.now().strftime("%Y-%m-%d"),120)
    if float(pred)>=float(adjusted_close):
        return render_template('answer_forecast_high.html', stock = stock.upper(), days = days,
                           pred = "{:.2f}".format(pred), adjusted_close = adjusted_close, data_graph = data_graph)
    else:
        return render_template('answer_forecast_low.html', stock=stock.upper(), days=days,
                               pred="{:.2f}".format(pred), adjusted_close=adjusted_close, data_graph = data_graph)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
