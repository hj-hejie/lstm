import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from matplotlib.pylab import date2num
import datetime
import tushare as ts

def plot(data):
    data_list=[]
    for date, row in data.iterrows():
        date_time=datetime.datetime.strptime(date, '%Y-%m-%d')
        t=date2num(date_time)
        o,h,c,l=row[:4]
        data_list.append((t,o,h,l,c));
    fig,ax=plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis_date()
    plt.xticks(rotation=45)
    plt.yticks()
    mpf.candlestick_ohlc(ax,data_list, colorup='r', colordown='green')
    plt.grid()
    plt.show()

if __name__=='__main__':
    data=ts.get_hist_data('600848', start='2017-07-01', end='2017-08-18')
    #plot(data)
    print(len(data['close']))
    print('hejie')
