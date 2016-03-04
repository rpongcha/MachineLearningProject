import Strategy.RBMRandomForestStrategy as rs
import Portfolio.RBMRandomForestPortfolio as rp
import models.random_forest as rf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import csv
import math

def trans_mean_std_save(p,filename):
    p=p.transpose()
    p.columns = ['Experiment '+str(i) for i in range(1,11)]
    p['Mean']=p.mean(axis =1)
    p['Standard Deviation']=p.std(axis =1)
    p.to_csv(os.path.join(os.path.expanduser('~'), 'RA','MachineLearningProject', 'data', 'random_forest',filename+'.csv'))
    
def back_testing_portfolio(symbol, data_point, capital, initial_margin=5000,maint_margin=3500,contract_size = 1000, purchase_size = 1):
    # Rolling for 10 times
    f1_score1_pd=pd.DataFrame()
    f1_score2_pd=pd.DataFrame()
    ClassReport_pd=pd.DataFrame()
    SharpeRatio_pd=pd.DataFrame()
    AnnualReturn_pd=pd.DataFrame()
    
    for i in range(10):
        # create RBMRandomForestStrategy object
        strategy = rs.RBMRandomForestStrategy(symbol, data_point,i)
        # get time stamp
        time_stamp = strategy.get_time_stamp()
        # get prices
        #prices = pd.DataFrame(strategy.get_prices(), columns=['prices'], index=time_stamp)
        prices = pd.DataFrame(strategy.get_prices(), columns=['prices'],index = time_stamp)

        # generate signals
        signals_pred, signals_test = strategy.generate_signals()
        f1_score1, f1_score2 = rf.print_f1_score(signals_test, signals_pred)
        class_report = rf.classification_error(signals_test, signals_pred)
        
        signals_pred = pd.DataFrame(signals_pred, columns=['signals'],index = time_stamp)
        signals_test = pd.DataFrame(signals_test, columns=['signals'],index = time_stamp)

        # create MarketOpenPortfolio object
        portfolio_pred = rp.RBMRandomForestPortfolio(symbol, prices, signals_pred, capital, initial_margin,maint_margin,contract_size, purchase_size)
        portfolio_test = rp.RBMRandomForestPortfolio(symbol, prices, signals_test, capital, initial_margin,maint_margin,contract_size, purchase_size)
    
        pred_port = portfolio_pred.backtest_portfolio()
        test_port = portfolio_test.backtest_portfolio()

        sharpe_pred = portfolio_pred.calculate_sharpe_ratio()
        sharpe_test = portfolio_test.calculate_sharpe_ratio()
    
        sharpe_ratio = pd.DataFrame({symbol+" Pred Sharpe Ratio":[sharpe_pred],symbol+" Test Sharpe Ratio: ":[sharpe_test]})
        annual_return = pd.DataFrame({symbol+" Pred Annualized Return":[pred_port['returns'].sum()*math.sqrt(float(252*24*60)/(12500*5))], symbol+" Test Sharpe Ratio: ":[test_port['returns'].sum()*math.sqrt(float(252*24*60)/(12500*5))]})
        
        f1_score1_pd=f1_score1_pd.append(f1_score1)
        f1_score2_pd=f1_score2_pd.append(f1_score2)
        ClassReport_pd=ClassReport_pd.append(class_report)
        SharpeRatio_pd=SharpeRatio_pd.append(sharpe_ratio)
        AnnualReturn_pd=AnnualReturn_pd.append(annual_return)
        
        prices_pd = pd.DataFrame({'signals_pred':signals_pred['signals'],'signals_test':signals_test['signals'],'prices':prices['prices']})
        prices_pd.to_csv(os.path.join(os.path.expanduser('~'), 'RA','MachineLearningProject', 'data', 'random_forest','price_'+symbol+'_'+str(i+1)+'.csv'))

        pred_port['Cumulative P&L'].plot(label='Predict')
        test_port['Cumulative P&L'].plot(label='Test')
 
        #plt.gcf().autofmt_xdate()
        plt.gcf().autofmt_xdate()
        plt.ylabel("Value ($)")
        plt.xlabel("TIme")
        plt.legend()
        plt.title(symbol+'_'+str(i+1)+" Cumulative P&L")
        plt.savefig(os.path.join(os.path.expanduser('~'), 'RA','MachineLearningProject', 'data', 'random_forest','Cum_P&L_'+symbol+'_'+str(i+1)+'.png'))
        plt.close()
        
        
        prices.plot()
        plt.gcf().autofmt_xdate()
        plt.ylabel("Value ($)")
        plt.xlabel("TIme")
        plt.legend()
        plt.title(symbol+'_'+str(i+1)+" Prices")
        plt.savefig(os.path.join(os.path.expanduser('~'), 'RA','MachineLearningProject', 'data', 'random_forest','Prices_'+symbol+'_'+str(i+1)+'.png'))
        plt.close()
        
    trans_mean_std_save(f1_score1_pd,symbol+' f1_score_report_1')
    trans_mean_std_save(f1_score2_pd,symbol+' f1_score_report_2')
    trans_mean_std_save(ClassReport_pd,symbol+' Classification_error')
    trans_mean_std_save(SharpeRatio_pd, symbol+' Sharpe_Ratio')
    trans_mean_std_save(AnnualReturn_pd, symbol+' Annulized_Return')
    
        # chart
        # calculate sharpe ratio
        # compare with real signals


if __name__ == '__main__':

    back_testing_portfolio('CL', 12500, 100000, 3850,3500,1000 )
    back_testing_portfolio('NG', 12500, 100000, 2090,1900,1000 )
    back_testing_portfolio('GC', 12500, 100000, 4675,4250,100 )
    back_testing_portfolio('PL', 12500, 100000, 2090,1900,50 )
    back_testing_portfolio('HG', 12500, 100000, 3135,2850,25000)
    back_testing_portfolio('ES', 12500, 100000, 5225,4750,50)
 

