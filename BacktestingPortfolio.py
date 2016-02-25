import Strategy.RBMRandomForestStrategy as rs
import Portfolio.RBMRandomForestPortfolio as rp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import datetime


def back_testing_portfolio(symbol, data_point, capital, initial_margin=5000, maint_margin=3500):
    # create RBMRandomForestStrategy object
    strategy = rs.RBMRandomForestStrategy(symbol, data_point)
    # get time stamp
    time_stamp = strategy.get_time_stamp()
    # for i in time_stamp[:5]:
    #    print i
    # get prices
    prices = pd.DataFrame(strategy.get_prices(), columns=['prices'])

    # generate signals
    signals = pd.DataFrame(strategy.generate_signals(), columns=['signals'])

    # create MarketOpenPortfolio object
    portfolio = rp.RBMRandomForestPortfolio(symbol, prices, signals, capital, initial_margin, maint_margin)
    # run back-testing
    my_portfolio = portfolio.backtest_portfolio()
    my_account = my_portfolio['portfolio'].values

    # plot chart
    datetimes = [datetime.datetime.strptime(t, "%Y/%m/%d %H:%M:%S.%f") for t in time_stamp]
    fig, my_chart = plt.subplots()
    my_chart.plot_date(datetimes, my_account, '-')
    hfmt = mdt.DateFormatter('%m/%d %H:%M')
    # my_chart.xaxis.set_major_locator(mdt.MinuteLocator())
    # my_chart.xaxis.set_major_formatter(hfmt)
    my_chart.fmt_xdata = mdt.DateFormatter("%H:%M:%S")
    fig.autofmt_xdate()
    plt.ylabel('portfolio value')
    plt.xlabel('time')
    # plt.xticks(rotation='vertical')
    plt.show()

    # calculate Sharpe Ratio
    sharpe_ratio = portfolio.calculate_sharpe_ratio()
    print('Annualized Sharpe Ratio : %.2f%%' % (sharpe_ratio*100))

    # compare with real signals


if __name__ == '__main__':
    back_testing_portfolio('ER', 12500, 100000, 5000)

