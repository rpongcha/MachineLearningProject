import Strategy.RBMRandomForestStrategy as rs
import Portfolio.RBMRandomForestPortfolio as rp
import pandas as pd


def back_testing_portfolio(symbol, data_point, capital, margin):
    # create RBMRandomForestStrategy object
    strategy = rs.RBMRandomForestStrategy(symbol, data_point)
    # get time stamp
    time_stamp = strategy.get_time_stamp()
    # get prices
    prices = pd.DataFrame(strategy.get_prices(), columns=['bars'], index=time_stamp)
    # generate signals
    signals = pd.DataFrame(strategy.generate_signals(), columns=['signal'], index=time_stamp)

    # create MarketOpenPortfolio object
    portfolio = rp.RBMRandomForestPortfolio(symbol, prices, signals, capital, margin)
    portfolio.backtest_portfolio()

    # chart
    # calculate sharpe ratio
    # compare with real signals


if __name__ == '__main__':
    back_testing_portfolio('ER', 12500, 100000, 5000)

