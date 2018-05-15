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
from keras.layers import LSTM, Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, AveragePooling1D, Flatten
from keras import regularizers
from keras.callbacks import EarlyStopping
#from keras.utils import plot_model
import pdb
import logging

logging.basicConfig(level=logging.INFO, format='%(filename)s[%(lineno)s] [%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HjLstm: 

	def __init__(self, pre_day, dict_day, stock_id, nn_layer):
		self.stock_id=stock_id
		self.nn_layer=nn_layer
		self.pre_day=pre_day
		self.dict_day=dict_day
		#self.split=0.8
		self.weights_file=self.stock_id+self.nn_layer+'_'+str(self.pre_day)+'_'+str(self.dict_day)+'.h5'
		self.data_file=self.stock_id+'.csv'
		#self.indexs={'close':{}, 'open':{}, 'high':{}, 'low':{}, 'volume':{}}
		self.indexs={'close':{}, 'volume':{}}
		for i in self.indexs:
			self.indexs[i]['scaler']=MinMaxScaler();
		#self.close_index=2
		#self.data_col_no=13

	def get_new_data(self):
		new_data=None;
		recent_date=max(self.data.index)
		recent_date=datetime.strptime(recent_date, '%Y-%m-%d')
		delta=timedelta(days=1)
		if recent_date  < datetime.now():
			recent_date=datetime.strftime(recent_date+delta, '%Y-%m-%d')
                        try:
                            new_data=ts.get_hist_data(self.stock_id, recent_date).sort_index(axis=0, ascending=True)
                        except IOError, e:
                            logger.info(e)
		return new_data

	def load_file(self, update=True):
		if os.path.exists(self.data_file):
			self.data=pd.read_csv(self.data_file, index_col='date')
			if update:
				new_data=self.get_new_data()
				if new_data is not None:
					self.data=self.data.append(new_data)
					self.data.to_csv(self.data_file)
			
		else:
			self.data=ts.get_hist_data(self.stock_id).sort_index(axis=0, ascending=True)
			self.data.to_csv(self.data_file)
		#self.data_col_no=self.data.columns.size
		#self.data_close=self.data['close']
		#self.data_open=self.data['open']
		#self.data_high=self.data['high']
		#self.data_low=self.data['low']
		#self.data_vol=self.data['volume']

	def load_data(self, update=True):
		self.load_file(update)
		seq_length=self.pre_day+self.dict_day
		for j in self.indexs:
			data=self.data[j]
			data=np.reshape(data, (-1, 1))
			data=self.indexs[j]['scaler'].fit_transform(data)
			data=np.reshape(data, len(data))
			reshaped_data = []
			for i in range(len(data) - seq_length+1):
				reshaped_data.append(data[i: i + seq_length])
			reshaped_data = np.array(reshaped_data)
                	setattr(self, 'train_x_'+j, reshaped_data[:, :self.pre_day])
			setattr(self, 'train_y_'+j, reshaped_data[:,-1])
			if not hasattr(self, 'train_all'):
				self.train_all=reshaped_data[:, :self.pre_day];
			else:
				self.train_all=np.concatenate((self.train_all, reshaped_data[:, :self.pre_day]), axis=1)
			#split = int(reshaped_data.shape[0] * self.split)
			#self.train_x = x[:split]
			#self.test_x = x[split:]
			#self.train_y = y[:split]
			#self.test_y = y[split:]

	def build_model(self):

		if self.nn_layer=='dnn1':
			self.model = Sequential()
            		self.model.add(Dense(100, input_dim=self.pre_day, activation='relu'))
			self.model.add(Dropout(0.5))
			self.model.add(Dense(70, activation='relu'))
			self.model.add(Dropout(0.5))
			self.model.add(Dense(30, activation='relu'))
                        self.model.add(Dropout(0.5))
			self.model.add(Dense(1, activation='linear'))

		elif self.nn_layer=='dnn2':
			self.model = Sequential()
            		self.model.add(Dense(512, input_dim=self.pre_day*len(self.indexs), activation='sigmoid'))
			self.model.add(Dropout(0.5))
            		self.model.add(Dense(128, input_dim=self.pre_day*len(self.indexs), activation='sigmoid'))
			self.model.add(Dropout(0.5))
			self.model.add(Dense(1, activation='sigmoid'))

		elif self.nn_layer=='conv5':
                	self.model=Sequential()
                	self.model.add(Conv1D(32, 6, activation='relu', input_shape=(None, 1)))
			self.model.add(AveragePooling1D(strides=1))
                        self.model.add(Dropout(0.5))
                        self.model.add(Conv1D(32, 6, activation='relu'))
			self.model.add(AveragePooling1D(strides=1))
                        self.model.add(Dropout(0.5))
                        self.model.add(Conv1D(32, 6, activation='relu'))
                        self.model.add(AveragePooling1D(strides=1))
                        self.model.add(Dropout(0.5))
                        self.model.add(Conv1D(32, 6, activation='relu'))
                        self.model.add(AveragePooling1D(strides=1))
                        self.model.add(Dropout(0.5))
                        self.model.add(Conv1D(32, 6, activation='relu'))
			self.model.add(GlobalAveragePooling1D())
                        self.model.add(Dropout(0.5))
                        self.model.add(Dense(1, activation='linear'))

		elif self.nn_layer=='lstm3':
			self.model=Sequential()
			self.model.add(LSTM(100, input_shape=(None, 1), return_sequences=True))
			self.model.add(LSTM(70, return_sequences=True))
			self.model.add(LSTM(30))
                        self.model.add(Dense(1, activation='sigmoid'))

		#self.model.compile(loss='mse', optimizer='rmsprop')
		self.model.compile(loss='msle', optimizer='nadam')
		#self.model.compile(loss='binary_crossentropy', optimizer='nadam')

		if(os.path.exists(self.weights_file)):
			self.model.load_weights(self.weights_file)
		#plot_model(self.model)

	def train_model(self, d=None):
		if(not hasattr(self, 'data')):
	                self.load_data()
		if(not hasattr(self, 'model')):
			self.build_model()

		#history=self.model.fit(self.train_all, self.train_y_close, batch_size=50, epochs=1000, validation_split=0.3, callbacks=[EarlyStopping('val_loss')])
		history=self.model.fit(self.train_all, self.train_y_close, batch_size=50, epochs=250)
		#history=self.model.fit(np.reshape(self.train_all, (len(self.train_all), -1, 1)), self.train_y_close, batch_size=50, epochs=10, validation_split=0.3)

		self.model.save_weights(self.weights_file)

		#plt.plot(history.history['loss'], label='loss')
		#plt.plot(history.history['val_loss'], label='val_loss')
		#plt.show()

		if d is not None:
			d[self.nn_layer+'loss']=history.history['loss']
			d[self.nn_layer+'val_loss']=history.history['val_loss']

	def predict(self, x=None):
		if not hasattr(self, 'data') and x is None:
                        self.load_data()
                if not hasattr(self, 'model'):
                	self.build_model()

		if x is None:
			self.predict_y=self.model.predict(self.train_all)
			#self.predict_y=self.model.predict(np.reshape(self.train_all, (len(self.train_all), -1, 1)))
		else:
			x_all=None
			for i in x:
				x_fit=self.indexs[i]['scaler'].transform(np.reshape(x[i], (-1, 1)))
				x_fit=np.reshape(x_fit, (1, -1))
				x_all= (x_fit if x_all is None else np.concatenate((x_all, x_fit), axis=1))
			predict_y=self.model.predict(x_all)
			#predict_y=self.model.predict(np.reshape(x_fit, (1, -1, 1)))
			return self.indexs['close']['scaler'].inverse_transform(predict_y)

	def plot(self):
		self.predict()
		predict_y_inverse = self.indexs['close']['scaler'].inverse_transform(self.predict_y)
		train_y_inverse = self.indexs['close']['scaler'].inverse_transform(np.reshape(self.train_y_close, (-1, 1)))
		plt.plot(predict_y_inverse, 'r-')
		plt.plot(predict_y_inverse, 'ro')
		plt.plot(train_y_inverse, 'go')
		plt.plot(train_y_inverse, 'g:')
		plt.show()

	def inverse_transform(self, y):
		return y*self.scaler.data_range_[self.close_index]+self.scaler.data_min_[self.close_index]
		

def do_train(lstm):
	lstm.train_model()

def train(lstms):
	for lstm in lstms:
		#threading.Thread(target=do_train, args=(lstm,)).start()
		do_train(lstm)

def predict(lstms, data):
	predict_ys=np.array([])
	for lstm in lstms:
		predict_y=lstm.predict(data)
		predict_ys=np.append(predict_ys, predict_y)
	return np.reshape(predict_ys, (len(predict_ys), 1))

def plot(lstms, data):
	predict_ys=predict(lstms, data)
	data_close=data[:, lstms[0].close_index]
	predict_data=np.append(data_close, predict_ys)
	new_data=lstms[0].get_new_data()
	if not new_data.empty:
		data_close=np.append(data_close, new_data['close'])
	plt.plot(np.reshape(data_close, (len(data_close), 1)), 'r.-')
	plt.plot(np.reshape(predict_data, (len(predict_data), 1)), 'g.:')
	plt.show()
	
def fortune(lstms, data):
	predict_ys=predict(lstms, data)
	data_close=data[:, lstms[0].close_index]
	predict_ys_flat=predict_ys.flatten()
	data_flat=np.repeat(data_close[-1], len(predict_ys_flat))
	rate=(predict_ys_flat-data_flat)/data_flat
	ask=np.repeat(0.7/(0.3-0.05), len(predict_ys_flat))
	new_data=lstms[0].get_new_data()
	real_data=None
        if not new_data.empty:
        	real_data=new_data['close']
	return data_flat, predict_ys_flat, rate, ask, rate>ask, real_data

def advise(lstm):
	lstm.load_data(False)
	data={}
	data_pre={}
	for i in lstm.indexs:
		data[i]=lstm.data[i][-lstm.pre_day:]
		data_pre[i]=lstm.data[i][-lstm.pre_day-1:-1]
	data_last=lstm.data['close'][-1]
	predict_last=lstm.predict(data_pre)[0][0]
	predict=lstm.predict(data)[0][0]
	predict_new=data_last*(1+(predict-predict_last)/predict_last)

	logger.info('\ndata_last:%s\n'\
		'predict_last:%s\n'\
		'predict:%s\n'\
		'predict/predict_last:%s\n'\
		'predict_new:%s\n'\
		'predict/data_last:%s\n'\
		'kelly:%s'\
		%(data_last,
			predict_last,
			predict,
			(predict-predict_last)/predict_last,
			predict_new,
			(predict-data_last)/data_last,
			0.5/(0.10+0.5/((predict-predict_last)/predict_last))))

	return predict_new
	
if __name__ == '__main__':
	#stock_id='600848'
	stock_id='600354'
	#stock_id='000001'
	#stock_id='hs300'
	#start='2011-01-01'
	#start='1990-01-01'
	#end='1990-01-05'
	#start='2018-02-11'
	#end='2018-02-23'
	#data_file=stock_id+'.csv'
	pre_day=64
	dict_day=1
	
	'''
	data=ts.get_hist_data(stock_id, start=start, end=end)
	data=data.sort_index(axis=0, ascending=True)
	print data
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
		
	#nn=HjLstm(pre_day, dict_day, stock_id, 'dnn_10_100_10_1')
	nn=HjLstm(pre_day, dict_day, stock_id, 'dnn2')
	#nn.load_file()
	nn.load_data(False)
	#nn.train_model()
	advise(nn)
        #nn.plot()
	#nn.load_data()
	#print nn.predict(nn.data.values[-pre_day:]).shape
		
	'''
	if len(sys.argv)>1:
		index=sys.argv[1]
		lstm=HjLstm(pre_day, int(index), stock_id, data)
		lstm.train_model()
		#lstm.plot()
	'''
	#else:
	'''
	lstms=[HjLstm(pre_day, i, stock_id, 'lstm3') for i in range(1, dict_day+1)]
	#train(lstms)
	lstms[0].load_data(False)
	data=lstms[0].data.values[-pre_day:]
	#print data
	plot(lstms, data)
		
	print 'hejie***************'
	prediction=fortune(lstms, data)
	for i in prediction:
		print i
	print 'hejie***************'
	'''
