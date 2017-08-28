import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

def load_data():
    #data=ts.get_hist_data('600848', start='2011-01-01', end='2018-01-18')
    #data.to_pickle('600848.pkl')
    data=pd.read_pickle(data_file)
    data=data['close']
    data=np.reshape(data, (len(data),1))
    data_all = np.array(data).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    x = reshaped_data[:, :-pre_count]
    y = reshaped_data[:, -pre_count:]
    y = np.reshape(y,(len(y),pre_count))
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]
    train_y = y[: split_boundary]
    test_y = y[split_boundary:]
    return train_x, train_y, test_x, test_y, scaler


def build_model():
    model = Sequential()
    model.add(LSTM(50, input_dim=1, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(pre_count))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    if(os.path.exists(weights_file)):
            model.load_weights(weights_file)
    return model


def train_model(model, train_x, train_y):
    model.fit(train_x, train_y, batch_size=1000, epochs=100, validation_split=0.1)
    model.save_weights(weights_file)
    return model

def superPredict(model, x, length):
    ys=[]
    for i in range(length):
        y=model.predict(x)
        ys.append(y[0])
        x=np.array([np.append(x[0][1:],y[0])])
        x=np.reshape(x, (x.shape[0], x.shape[1],1))
    return ys;

def plotPredict(test_y, predict_y):
    fig=plt.figure(1)
    doPlotPredict(test_y, 'r-')
    doPlotPredict(predict_y, 'g:')
    plt.show()

def doPlotPredict(ys, color):
    for i in range(len(ys)):
        plt.plot(np.append([None for x in range(i)], ys[i]), color)

data_file='600848.pkl'
#weights_file='600848_50_100.h5'
#weights_file='600848_50_100_200.h5'
#weights_file='600848_50_100_3.h5'
weights_file='600848_50_100_50.h5'
span=0
pre_count=50
sequence_length=100
split=0.8

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, scaler = load_data()
    #train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    #test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    model=build_model()
    model=train_model(model, train_x, train_y)
    predict_y=model.predict(test_x[span:])
    #predict_y2=superPredict(model, test_x[span:span+1], 50);
    predict_y = scaler.inverse_transform(predict_y)
    #predict_y2 = scaler.inverse_transform(predict_y2)
    test_y = scaler.inverse_transform(test_y)
    #fig2 = plt.figure(1)
    #plt.plot(predict_y, 'g:')
    #plt.plot(predict_y2, 'b:')
    #plt.plot(test_y[span:], 'r-')
    #plt.show()
    plotPredict(test_y, predict_y)
