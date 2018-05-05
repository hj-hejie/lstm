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
from neupy import algorithms, estimators
import pdb
import logging

logging.basicConfig(level=logging.INFO, format='%(filename)s[%(lineno)s] [%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HjGrnn: 

	def __init__(self, pre_day, dict_day, stock_id):
		self.stock_id=stock_id
		self.pre_day=pre_day
		self.dict_day=dict_day
		self.split=0.3
		self.data_file=self.stock_id+'.csv'
		self.indexs={'close':{}}
		#self.indexs={'close':{}, 'open':{}, 'high':{}, 'low':{}, 'volume':{}}
		for i in self.indexs:
			self.indexs[i]['scaler']=MinMaxScaler();

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
                self.train_x, self.test_x, self.train_y, self.test_y=train_test_split(self.train_all, self.train_y_close, test_size=self.split)

	def build_model(self):
                self.model=algorithms.GRNN(std=0.1)
                self.model.train(self.train_x, self.train_y)

	def predict(self, xs=None):
		if not hasattr(self, 'data') and xs is None:
                        self.load_data()
                if not hasattr(self, 'model'):
                	self.build_model()

		if xs is None:
			self.predict_y=self.model.predict(self.test_x)
                        return estimators.rmse(self.predict_y, self.test_y)
		else:
                        x_alls=[]
                        for x in xs:
			    x_all=None
			    for i in x:
				x_fit=self.indexs[i]['scaler'].transform(np.reshape(x[i], (-1, 1)))
				x_fit=np.reshape(x_fit, (1, -1))
				x_all= (x_fit if x_all is None else np.concatenate((x_all, x_fit), axis=1))

                            x_alls.append(np.reshape(x_all, (-1,)))

			predict_y=self.model.predict(x_alls)
			return self.indexs['close']['scaler'].inverse_transform(predict_y)

	def plot(self):
		self.predict()
		predict_y_inverse = self.indexs['close']['scaler'].inverse_transform(self.predict_y)
		test_y_inverse = self.indexs['close']['scaler'].inverse_transform(np.reshape(self.test_y, (-1, 1)))
		plt.plot(predict_y_inverse, 'g:')
		plt.plot(test_y_inverse, 'r-')
		plt.show()

def advise(lstm):
	lstm.load_data(False)
	data={}
	data_pre={}
	for i in lstm.indexs:
		data[i]=lstm.data[i][-lstm.pre_day:]
		data_pre[i]=lstm.data[i][-lstm.pre_day-1:-1]
	data_last=lstm.data['close'][-1]
        predicts=lstm.predict([data_pre, data])
	predict_last=predicts[0][0]
	predict=predicts[1][0]
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
	pre_day=30
	dict_day=1
	
	nn=HjGrnn(pre_day, dict_day, stock_id)
	advise(nn)
	#nn.load_file()
	#nn.load_data(False)
        #print nn.predict()
        #nn.plot()
