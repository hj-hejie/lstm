import os
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

class HjLstm: 

	def __init__(self, pre_day, dict_day, stock_id, data):
                self.data=data
		self.stock_id=stock_id
		self.nn_layer='_50_100'
		self.pre_day=pre_day
		self.dict_day=dict_day
		self.split=0.8
		#self.weights_file='600848_50_100.h5'
		self.weights_file=self.stock_id+self.nn_layer+'_'+str(self.pre_day)+'_'+str(self.dict_day)+'.h5'
		self.build_model()
		self.load_data()
		
		
	def load_data(self):
		sequence_length=self.pre_day+self.dict_day
		data=self.data['close']
		data=np.reshape(data, (len(data),1))
		data_all = np.array(data).astype(float)
		self.scaler = MinMaxScaler()
		data_all = self.scaler.fit_transform(data_all)
		data = []
		for i in range(len(data_all) - sequence_length):
			data.append(data_all[i: i + sequence_length])
		reshaped_data = np.array(data).astype('float64')
		x = reshaped_data[:, :-self.dict_day]
		y = reshaped_data[:, -1]
		split_boundary = int(reshaped_data.shape[0] * self.split)
		self.train_x = x[: split_boundary]
		self.test_x = x[split_boundary:]
		self.train_y = y[: split_boundary]
		self.test_y = y[split_boundary:]


	def build_model(self):
		self.model = Sequential()
		self.model.add(LSTM(50, input_shape=(None, 1), return_sequences=True))
		self.model.add(LSTM(100))
		self.model.add(Dense(1))
		self.model.add(Activation('linear'))
		self.model.compile(loss='mse', optimizer='rmsprop')
		if(os.path.exists(self.weights_file)):
				self.model.load_weights(self.weights_file)


	def train_model(self):
		self.model.fit(self.train_x, self.train_y, batch_size=512, epochs=1, validation_split=0.1)
		self.model.save_weights(self.weights_file)
		
	def do_predict(self):
		self.predict_y=self.model.predict(self.test_x)
		
	def plot(self):
                self.do_predict()
		predict_y_inverse = self.scaler.inverse_transform(self.predict_y)
		test_y_inverse = self.scaler.inverse_transform(self.test_y)
		plt.figure(1)
		plt.plot(predict_y_inverse, 'g:')
		plt.plot(test_y_inverse, 'r-')
		plt.show()

        def predict(self, test_x):
                predict_y=self.model.predict(test_x)
                return self.scaler.inverse_transform(predict_y)


def do_train(lstm):
	lstm.train_model()

def train(lstms):
	for lstm in lstms:
		#threading.Thread(target=do_train, args=(lstm,)).start()
                do_train(lstm)

def predict(lstms, test_x):
    predict_ys=np.arrays([])
    for lstm in lstms:
        predict_y=lstm.predict(test_x)
        predict_y=np.append(predict_ys, predict_y);

def plot(data, lstms, test_x):
    predict_y=predict(lstms, test_x)
    plt.figure(1)
    plt.plot(np.append(test_x, predict_y))

        		
if __name__ == '__main__':
        stock_id='600848'
        data_file=stock_id+'.pkl'

        #data=ts.get_hist_data('600848', start='2011-01-01', end='2018-01-18')
        #data.to_pickle('600848.pkl')
	data=pd.read_pickle(data_file)

        #lstm=HjLstm(10, 1)
	lstm10_2=HjLstm(10, 2, stock_id, data)
	lstm10_3=HjLstm(10, 3, stock_id, data)
	lstm10_4=HjLstm(10, 4, stock_id, data)
	lstm10_5=HjLstm(10, 5, stock_id, data)
	lstm10_6=HjLstm(10, 6, stock_id, data)
	lstm10_7=HjLstm(10, 7, stock_id, data)

	lstms=[lstm10_2,lstm10_3,lstm10_4,lstm10_5,lstm10_6,lstm10_7]

        #train(lstms)


