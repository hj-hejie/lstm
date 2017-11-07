#!/usr/bin/python

import sys
import os
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.utils import plot_model

class HjLstm: 

	def __init__(self, pre_day, dict_day, stock_id, data):
		self.data=data
		self.stock_id=stock_id
		self.nn_layer='_50_100'
		self.pre_day=pre_day
		self.dict_day=dict_day
		self.split=0.8
		self.weights_file=self.stock_id+self.nn_layer+'_'+str(self.pre_day)+'_'+str(self.dict_day)+'.h5'
		self.scaler = MinMaxScaler()
		self.build_model()
		self.load_data()

	def load_data(self):
		seq_length=self.pre_day+self.dict_day
		data=self.data
		data=np.reshape(data, (len(data),1))
		data= self.scaler.fit_transform(data)
		reshaped_data = []
		for i in range(len(data) - seq_length):
			reshaped_data.append(data[i: i + seq_length])
		reshaped_data = np.array(reshaped_data)
		x = reshaped_data[:, :-self.dict_day]
		y = reshaped_data[:, -1]
		split = int(reshaped_data.shape[0] * self.split)
		self.train_x = x[: split]
		self.test_x = x[split:]
		self.train_y = y[: split]
		self.test_y = y[split:]

	def build_model(self):
		self.model = Sequential()
		self.model.add(LSTM(50, input_shape=(None, 1), return_sequences=True))
		self.model.add(LSTM(100))
		self.model.add(Dense(1))
		self.model.add(Activation('linear'))
		self.model.compile(loss='mse', optimizer='rmsprop')
		if(os.path.exists(self.weights_file)):
				self.model.load_weights(self.weights_file)
		plot_model(self.model)

	def train_model(self):
		history=self.model.fit(self.train_x, self.train_y, batch_size=20, epochs=2, validation_split=0.3)
		self.model.save_weights(self.weights_file)
		
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.show()

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
	predict_ys=np.array([])
	for lstm in lstms:
		predict_y=lstm.predict(test_x)
		predict_ys=np.append(predict_ys, predict_y);
	return np.reshape(predict_ys, (len(predict_ys), 1))

def plot(lstms, data):
	split = int(len(data) * 0.8)
	test_data=data[split:]
	test_x=test_data[0:10]
	test_x=np.reshape(test_x, (1, len(test_x), 1))
	predict_y=predict(lstms, test_x)
	predict_xy=np.append(test_x, predict_y)
	plt.figure(1)
	plt.plot(np.reshape(test_data, (len(test_data), 1)), 'r-')
	plt.plot(np.reshape(predict_xy, (len(predict_xy), 1)), 'g:')
	plt.show()

if __name__ == '__main__':
	stock_id='600848'
	#start='2011-01-01'
	start='1990-01-01'
	end='2018-01-18'
	data_file=stock_id+'_'+start+'_'+end+'.pkl'
	pre_day=20

	#data=ts.get_hist_data(stock_id, start=start, end=end)
	#data.to_pickle(data_file)
	data=pd.read_pickle(data_file)['close']
	#print np.array(data).shape
	
	if len(sys.argv)>1:
		index=sys.argv[1]
		lstm=HjLstm(pre_day, int(index), stock_id, data)
		lstm.train_model()
		#lstm.plot()
	
	'''
	else:
		lstms=[HjLstm(pre_day, i, stock_id, data) for i in range(1,8)]
		train(lstms)
		plot(lstms, data)
	'''

