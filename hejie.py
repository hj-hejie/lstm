#!/usr/bin/python

import sys
import os
from datetime import datetime, timedelta
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import tushare as ts
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
#from keras.utils import plot_model
import pdb

class HjLstm: 

	def __init__(self, pre_day, dict_day, stock_id, nn_layer='_50_100'):
		self.stock_id=stock_id
		self.nn_layer=nn_layer
		self.pre_day=pre_day
		self.dict_day=dict_day
		self.split=0.8
		self.weights_file=self.stock_id+self.nn_layer+'_'+str(self.pre_day)+'_'+str(self.dict_day)+'.h5'
		self.data_file=self.stock_id+'.csv'
		self.scaler = MinMaxScaler()
		#self.build_model()
		#self.load_data()
		
	def load_file(self):
		if(os.path.exists(self.data_file)):
			self.data=pd.read_csv(self.data_file, index_col='date')
			recent_date=max(self.data.index)
			recent_date=datetime.strptime(recent_date, '%Y-%m-%d')
			delta=timedelta(days=1)
			if recent_date  < datetime.now():
				recent_date=datetime.strftime(recent_date+delta, '%Y-%m-%d')
				new_data=ts.get_hist_data(self.stock_id, recent_date)
				self.data=new_data.append(self.data)
			
		else:
			self.data=ts.get_hist_data(self.stock_id)

		self.data.to_csv(self.data_file)

	def load_data(self):
		self.load_file()
		seq_length=self.pre_day+self.dict_day
		data=self.data['close']
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

		if self.nn_layer=='_50_100':
			self.model = Sequential()
			self.model.add(LSTM(50, input_shape=(None, 1), return_sequences=True))
			self.model.add(LSTM(100))
			self.model.add(Dense(1, activation='linear'))

		elif self.nn_layer=='nn_10_100_10_1':
			self.model=Sequential()
			self.model.add(LSTM(50, input_shape=(None, 1), return_sequences=True))
			self.model.add(LSTM(100))
                        self.model.add(Dense(1, activation='linear'))		
			
			
		elif self.nn_layer=='xdnn_10_100_10_1':
			self.model = Sequential()
            		self.model.add(Dense(100, input_shape=(self.pre_day,), activation='relu'))
			self.model.add(Dropout(0.5))
			self.model.add(Dense(10, activation='relu'))
			self.model.add(Dropout(0.5))	
			self.model.add(Dense(1, activation='linear'))

		elif self.nn_layer=='dnn_10_100_10_1':
			self.model=Sequential()
			self.model.add(Conv1D(5, 5, activation='relu', input_shape=(None, 1)))
                        self.model.add(MaxPooling1D())
                        self.model.add(Conv1D(5, 5, activation='relu'))
			self.model.add(GlobalAveragePooling1D())
			self.model.add(Dense(1, activation='sigmoid'))

		self.model.compile(loss='mse', optimizer='rmsprop')

		if(os.path.exists(self.weights_file)):
			self.model.load_weights(self.weights_file)
		#plot_model(self.model)

	def train_model(self, d=None):

		if(not hasattr(self, 'data')):
	                self.load_data()
		if(not hasattr(self, 'model')):
			self.build_model()

		if type(self.model.get_layer(index=1)) is Dense:
			self.train_x=np.reshape(self.train_x, self.train_x.shape[:-1])
		history=self.model.fit(self.train_x, self.train_y, batch_size=50, epochs=2000, validation_split=0.3)

		self.model.save_weights(self.weights_file)

		#plt.plot(history.history['loss'], label='loss')
		#plt.plot(history.history['val_loss'], label='val_loss')
		#plt.show()

		if d is not None:
			d[self.nn_layer+'loss']=history.history['loss']
			d[self.nn_layer+'val_loss']=history.history['val_loss']

	def predict(self, x=None):

		if(not hasattr(self, 'data')):
                        self.load_data()
                if(not hasattr(self, 'model')):
                	self.build_model()

		if x is None:
			if type(self.model.get_layer(index=1)) is Dense:
				self.test_x=np.reshape(self.test_x, self.test_x.shape[:-1])
			self.predict_y=self.model.predict(self.test_x)
		else:
			predict_y=self.model.predict(test_x)
			return self.scaler.inverse_transform(predict_y)

	def plot(self):

		self.predict()
		predict_y_inverse = self.scaler.inverse_transform(self.predict_y)
		test_y_inverse = self.scaler.inverse_transform(self.test_y)
		plt.plot(predict_y_inverse, 'g:')
		plt.plot(test_y_inverse, 'r-')
		plt.show()
		

def do_train(lstm):
	lstm.train_model()

def train(lstms):
	for lstm in lstms:
		#threading.Thread(target=do_train, args=(lstm,)).start()
		do_train(lstm)

def predict(lstms, data):
	data_x=np.reshape(data, (1, len(test_x), 1))
	predict_ys=np.array([])
	for lstm in lstms:
		predict_y=lstm.predict(data_x)
		predict_ys=np.append(predict_ys, predict_y);
	return np.reshape(predict_ys, (len(predict_ys), 1))

def plot(lstms, data):
	predict_ys=predict(lstms, data)
	predict_data=np.append(data, predict_ys)
	plt.plot(np.reshape(data, (len(data), 1)), 'r-')
	plt.plot(np.reshape(predict_data, (len(predict_data), 1)), 'g:')
	plt.show()
	
def fortune(lstms, data):
	predict_ys=predict(lstms, data)
	predict_ys_flat=predict_ys.flatten()
	data_falt=np.repeat(data[-1], len(predict_ys_flat))
	rate=(predict_ys_flat-data_falt)/data_falt
	ask=0.3/(0.7-0.1)
	return rate>ask
	
if __name__ == '__main__':
	#stock_id='600848'
	#stock_id='000001'
	stock_id='hs300'
	#start='2011-01-01'
	start='1990-01-01'
	#end='1990-01-05'
	#start='2018-02-18'
	end='2018-02-23'
	#data_file=stock_id+'.csv'
	pre_day=20
	dict_day=7
	
	'''
	data=ts.get_hist_data(stock_id, start=start, end=end)
	#data.to_pickle(data_file)
	#data=pd.read_pickle(data_file)['close']
	data=data['close']
	#data.to_csv(data_file)
	print 'hejie*****************'
	print np.array(data).shape
	print np.array(data)
	print 'hejie---------------------'
	'''

	'''
	mgr=mp.Manager()
	d=mgr.dict()
	hj1=HjLstm(pre_day, dict_day, stock_id, data, 'hj1')
	hj2=HjLstm(pre_day, dict_day, stock_id, data, 'hj2')
	
	hj1_p=mp.Process(target=hj1.train_model, args=(d,))
	hj2_p=mp.Process(target=hj2.train_model, args=(d,))
	
	hj1_p.start()
	hj2_p.start()
	
	hj1_p.join()
	hj2_p.join()

	for i,v in d.items():
		plt.plot(v, label=i)

	plt.show()
	'''
	
	nn=HjLstm(pre_day, dict_day, stock_id, 'dnn_10_100_10_1')
	data=lstms[0].data['close'][-5:]
	#nn.load_file()
	#nn.train_model()
        #nn.plot()
	#print nn.test_y.shape
	#print nn.model.predict(np.reshape(nn.test_x, nn.test_x.shape[:-1])).shape
		
	'''
	if len(sys.argv)>1:
		index=sys.argv[1]
		lstm=HjLstm(pre_day, int(index), stock_id, data)
		lstm.train_model()
		#lstm.plot()
	'''
	'''
	#else:
		lstms=[HjLstm(pre_day, i, stock_id, data) for i in range(1,8)]
		train(lstms)
		data=lstms[0].data['close'][-5:]
		print 'hejie***************'
		print data
		#plot(lstms, data)
		#print fortune(lstms, data)
	'''

