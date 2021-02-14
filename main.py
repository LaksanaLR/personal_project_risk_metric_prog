import numpy as np
import pandas as pd
import pandas_datareader as web
import statsmodels.api as sm
import sklearn
from statsmodels import regression
from sklearn.linear_model import LinearRegression

print('Welcome, choose your option \n- Sortino Ratio\n- Calmar Ratio\n- MAR\n- Sharpe Ratio\n- Beta\n- Alpha\n- CAPM\n- R-Squared\n- Stdev')


def myCommand():
    query = input('What do you want to measure? ')
    return query


# Creating Option menu
if __name__ == '__main__':
    while True:
        query = myCommand()
        query = query.lower()

        if 'sortino ratio' and 'sortino' in query:
            x = [float(input("Gains/loss: ")) for i in range(int(input("Number of years since inception : ")))]
            w = float(input("What is minimum accepted return?: "))
            y = np.array(x)
            k = np.array(w)
            z = y - k
            df = pd.DataFrame(z)
            downside_total = np.sqrt(df ** 2)
            downside_dev = downside_total.mean()
            returns = input('Type in the target return or real return of portfolio/fund in terms of percentage (exclude percentage sign): ')
            rfr = 0.09
            sortino = (float(returns) - float(rfr)) / float(downside_dev)
            print(sortino)
        elif 'calmar ratio' and 'calmar' in query:
            x = [float(input("Annual Gains/loss (exclude percentage sign): ")) for i in range(3)]
            y = np.array(x)
            df = pd.DataFrame(y)
            first_val = df.head(1)
            first_valfl = first_val.iloc[0]
            last_val = df.tail(1)
            last_valfl = last_val.iloc[0]
            fl_valdf = first_valfl / last_valfl
            change = df.pct_change()
            max_drawdown = change.min()
            cagr = (float(fl_valdf) ** (1 / len(y)) - 1) * 100
            calmar = cagr / max_drawdown
            print(calmar)
        elif 'mar' in query:
            x = [float(input("Gains/loss (exclude percentage sign): ")) for i in range(int(input("Number of years since inception (must be more than 1): ")))]
            y = np.array(x)
            df = pd.DataFrame(y)
            first_val = df.head(1)
            first_valfl = first_val.iloc[0]
            last_val = df.tail(1)
            last_valfl = last_val.iloc[0]
            fl_valdf = first_valfl/last_valfl
            change = df.pct_change()
            max_drawdown = change.min()
            cagr = (float(fl_valdf) ** (1 / len(y)) - 1) * 100
            mar = cagr / max_drawdown
            print(mar)
        elif 'sharpe ratio' and 'sharpe' in query:
            x = [float(input("Gains/loss: ")) for i in range(int(input("Number of years since inception : ")))]
            y = np.array(x)
            std = np.std(y)
            returns = input('Type in the target return or real return of portfolio/fund in terms of percentage (exclude percentage sign): ')
            rfr = input('Type in the risk-free rate: ')
            sharpe = (float(returns) - float(rfr)) / float(std)
            print(sharpe)
        elif 'beta' in query:
            stock = str(input("Ticker: "))
            benchmark = str(input("Benchmark Index: "))
            startp = input("Initial Date (please use YYYY-MM-DD format): ")
            stock_data = web.DataReader(stock, 'yahoo', startp)['Adj Close']
            stock_returns = np.log(1 + stock_data.pct_change())[1:]
            stock_returns = stock_returns.dropna()
            stock_returns = np.array(stock_returns)
            benchmark_data = web.DataReader(benchmark, 'yahoo', startp)['Adj Close']
            benchmark_ret = np.log(1 + benchmark_data.pct_change())[1:]
            benchmark_ret = benchmark_ret.dropna()
            benchmark_ret = np.array(benchmark_ret)

            def linear_regression(x, y):
                x = sm.add_constant(x)
                linreg = regression.linear_model.OLS(y, x).fit()
                return linreg.params[1]
            print("Beta: ", linear_regression(benchmark_ret, stock_returns))
        elif 'alpha' in query:
            stock = str(input("Ticker: "))
            benchmark = str(input("Benchmark Index: "))
            startp = input("Initial Date (please use YYYY-MM-DD format): ")
            stock_data = web.DataReader(stock, 'yahoo', startp)['Adj Close']
            stock_returns = np.log(1 + stock_data.pct_change())[1:]
            stock_returns = stock_returns.dropna()
            stock_returns = np.array(stock_returns)
            benchmark_data = web.DataReader(benchmark, 'yahoo', startp)['Adj Close']
            benchmark_ret = np.log(1 + benchmark_data.pct_change())[1:]
            benchmark_ret = benchmark_ret.dropna()
            benchmark_ret = np.array(benchmark_ret)

            def linear_regression(x, y):
                x = sm.add_constant(x)
                linreg = regression.linear_model.OLS(y, x).fit()
                return linreg.params[0]
            print("Alpha: ", linear_regression(benchmark_ret, stock_returns))
        elif 'capm' in query:
            stock = str(input("Ticker: "))
            benchmark = str(input("Benchmark Index: "))
            rfr = float(input("Risk Free Rate: "))
            startp = input("Initial Date (please use YYYY-MM-DD format): ")
            stock_data = web.DataReader(stock, 'yahoo', startp)['Adj Close']
            stock_returns = np.log(1 + stock_data.pct_change())[1:]
            stock_returns = stock_returns.dropna()
            stock_returns = np.array(stock_returns)
            benchmark_data = web.DataReader(benchmark, 'yahoo', startp)['Adj Close']
            benchmark_ret_log = np.log(1 + benchmark_data.pct_change())[1:]
            benchmark_ret_log = benchmark_ret_log.dropna()
            benchmark_ret_log = np.array(benchmark_ret_log)
            startd = benchmark_data.index.min()
            endd = benchmark_data.index.max()
            in_price = benchmark_data.loc[startd]
            fin_price = benchmark_data.loc[endd]
            mark_ret = ((fin_price-in_price)/in_price)*100

            def linear_regression(x, y):
                x = sm.add_constant(x)
                linreg = regression.linear_model.OLS(y, x).fit()
                return linreg.params[1]
            beta = linear_regression(benchmark_ret_log, stock_returns)
            capm = rfr + (beta*(mark_ret-rfr))
            print(capm)
        elif 'r-squared' in query:
            stock = str(input("Ticker: "))
            benchmark = str(input("Benchmark Index: "))
            startp = input("Initial Date (please use YYYY-MM-DD format): ")
            stock_data = web.DataReader(stock, 'yahoo', startp)['Adj Close']
            stock_data = np.array(stock_data)
            benchmark_data = web.DataReader(benchmark, 'yahoo', startp)['Adj Close']
            benchmark_data = np.array(benchmark_data).reshape(-1, 1)
            model = LinearRegression()
            model.fit(benchmark_data, stock_data)
            r_value = model.score(benchmark_data, stock_data)
            r_squared = r_value**2
            print(r_squared)
        elif 'standard deviation' and 'stdev' in query:
            ticker = str(input("Ticker: "))
            startp = input("Initial Date (please use YYYY-MM-DD format): ")
            data = web.DataReader(ticker, 'yahoo', startp)['Adj Close']
            data = np.array(data)
            std = data.std()
            print(std)
        else:
            print('Error!: No Option with the name "', query,'"')
