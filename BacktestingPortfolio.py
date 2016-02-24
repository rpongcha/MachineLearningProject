import Strategy.RBMRandomForestStrategy as rs
import Portfolio.RBMRandomForestPortfolio as rp
import pandas as pd


def back_testing_portfolio(symbol, data_point, capital, initial_margin=5000, maint_margin=3500):
    # create RBMRandomForestStrategy object
    strategy = rs.RBMRandomForestStrategy(symbol, data_point)
    # get time stamp
    time_stamp = strategy.get_time_stamp()
    for i in time_stamp[:5]:
        print i
    # get prices
    prices = pd.DataFrame(strategy.get_prices(), columns=['prices'])

    # generate signals
    signals = pd.DataFrame(strategy.generate_signals(), columns=['signals'])

    # create MarketOpenPortfolio object
    portfolio = rp.RBMRandomForestPortfolio(symbol, prices, signals, capital, initial_margin, maint_margin)
    # run back-testing
    portfolio.backtest_portfolio()

    # chart
    # calculate sharpe ratio
    # compare with real signals


if __name__ == '__main__':
    back_testing_portfolio('ER', 12500, 100000, 5000)

