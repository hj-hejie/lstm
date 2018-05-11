#!/usr/bin/python


import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import tushare as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import Constant
from keras import backend as K
import pdb
import logging

logging.basicConfig(level=logging.INFO, format='%(filename)s[%(lineno)s] [%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class RBFLayer(Layer):

	def __init__(self, output_dim, train_x, **kwargs):
		self.output_dim=output_dim
		self.means=KMeans(n_clusters=output_dim, random_state=0).fit(train_x).cluster_centers_
		self.sigmas=np.array([(np.sum([np.linalg.norm(j-i)**2 for j in self.means])/len(self.means))**0.5 for i in self.means])
		super(RBFLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.means_K=self.add_weight(name='means', shape=(self.output_dim, input_shape[1]), initializer=Constant(self.means), trainable=False)
		self.sigmas_K=self.add_weight(name='sigmas', shape=(self.output_dim,), initializer=Constant(self.sigmas), trainable=False)
		super(RBFLayer, self).build(input_shape)	

	def call(self, x):
		C = K.expand_dims(self.means_K)
		H = K.transpose(C-K.transpose(x))
		return K.exp(-K.sum(H**2, axis=1))

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

class HjRbf: 

	def __init__(self, pre_day, dict_day, stock_id):
		self.stock_id=stock_id
		self.pre_day=pre_day
		self.dict_day=dict_day
		self.split=1
		self.weights_file=self.stock_id+'_rbf_'+str(self.pre_day)+'_'+str(self.dict_day)+'.h5'
		self.data_file=self.stock_id+'.csv'
		#self.indexs={'close':{}, 'open':{}, 'high':{}, 'low':{}, 'volume':{}}
		self.indexs={'close':{}}
		for i in self.indexs:
			self.indexs[i]['scaler']=MinMaxScaler()

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

	def load_data(self, update=True):
		if(not hasattr(self, 'data')):
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
			#self.train_x, self.test_x, self.train_y, self.test_y=train_test_split(self.train_all, self.train_y_close, test_size=1-self.split)
			split=int(len(self.train_all)*self.split)
			self.train_x=self.train_all[:split]
			self.test_x=self.train_all[split:]
			self.train_y=self.train_y_close[:split]
			self.test_y=self.train_y_close[split:]

	def build_model(self):
		self.model = Sequential()
		self.model.add(RBFLayer(256, self.train_x, input_shape=(self.pre_day,)))
		self.model.add(Dense(1))
		self.model.compile(loss='mse', optimizer='nadam')
		
		if(os.path.exists(self.weights_file)):
			self.model.load_weights(self.weights_file)

	def train_model(self):
		if(not hasattr(self, 'data')):
			self.load_data()
		if(not hasattr(self, 'model')):
			self.build_model()	
		self.model.fit(self.train_x, self.train_y, epochs=300000)

		self.model.save_weights(self.weights_file)

	def predict(self, x=None):
		if not hasattr(self, 'data') and x is None:
                        self.load_data()
                if not hasattr(self, 'model'):
                	self.build_model()

		if x is None:
			self.predict_y=self.model.predict(self.train_x)
		else:
			x_all=None
			for i in x:
				x_fit=self.indexs[i]['scaler'].transform(np.reshape(x[i], (-1, 1)))
				x_fit=np.reshape(x_fit, (1, -1))
				x_all= (x_fit if x_all is None else np.concatenate((x_all, x_fit), axis=1))

			predict_y=self.model.predict(x_all)
			return self.indexs['close']['scaler'].inverse_transform(predict_y)

	def plot(self):
		self.predict()
		predict_y_inverse = self.indexs['close']['scaler'].inverse_transform(self.predict_y)
		test_y_inverse = self.indexs['close']['scaler'].inverse_transform(np.reshape(self.train_y, (-1, 1)))
		plt.plot(predict_y_inverse, 'r-')
		plt.plot(predict_y_inverse, 'ro')
		plt.plot(test_y_inverse, 'go')
		plt.plot(test_y_inverse, 'g:')
		plt.show()

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
	
	nn=HjRbf(pre_day, dict_day, stock_id)
	#nn.load_file()
	nn.load_data(False)
	nn.train_model()
        #advise(nn)
        #nn.plot()
