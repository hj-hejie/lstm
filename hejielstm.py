import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

def load_data(sequence_length=10, split=0.8):
    #data=ts.get_hist_data('600848', start='2011-01-01', end='2018-01-18')
    #data.to_pickle('600848.pkl')
    data=pd.read_pickle('600848.pkl')
    data=data['close']
    data=np.reshape(data, (len(data),1))
    data_all = np.array(data).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]
    train_y = y[: split_boundary]
    test_y = y[split_boundary:]
    return train_x, train_y, test_x, test_y, scaler


def build_model():
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.load_weights('hjmodel.h5')
    return model


def train_model(model, train_x, train_y):
    model.fit(train_x, train_y, batch_size=512, nb_epoch=100, validation_split=0.1)
    model.save_weights('hjmodel.h5')
    return model

def superPredict(x):
    ys=[]
    y=model.predict(x)
    ys.append(y)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, scaler = load_data()
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    model=build_model()
    #model=train_model(model, train_x, train_y)
    predict_y=model.predict(test_x)
    predict_y = scaler.inverse_transform(predict_y)
    test_y = scaler.inverse_transform(test_y)
    fig2 = plt.figure(2)
    plt.plot(predict_y, 'g:')
    plt.plot(test_y, 'r-')
    plt.show()

