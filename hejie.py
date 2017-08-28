import os
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

class HjLstm:

	def __init__(self, pre_day, dict_day):
		self.stock_id='600848'
		self.nn_layer='_50_100'
		self.pre_day=pre_day
		self.dict_day=dict_day
		self.split=0.8
		#self.weights_file='600848_50_100.h5'
		self.data_file=self.stock_id+'.pkl'
		self.weights_file=self.stock_id+self.nn_layer+'_'+str(self.pre_day)+'_'+str(self.dict_day)+'.h5'
		build_model(self)
		load_data(self)
		
		
	def load_data(self):
		#data=ts.get_hist_data('600848', start='2011-01-01', end='2018-01-18')
		#data.to_pickle('600848.pkl')
		sequence_length=self.pre_day+self.dict_day
		data=pd.read_pickle(self.data_file)
		data=data['close']
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
		
	def predict(self):
		self.predict_y=self.model.predict(self.test_x)
		
	def plot(self):
		predict_y_inverse = self.scaler.inverse_transform(self.predict_y)
		test_y_inverse = self.scaler.inverse_transform(self.test_y)
		plt.figure(1)
		plt.plot(predict_y_inverse, 'g:')
		plt.plot(test_y_inverse, 'r-')
		plt.show()

def do_train(lstm):
	lstm.train_model()

def train(lstms):
	for lstm in lstms:
		threading.Thread(target=do_train, args=(lstm,)).start()
		
if __name__ == '__main__':
	#lstm=HjLstm(10, 1)
	lstm10_2=HjLstm(10, 2)
	lstm10_3=HjLstm(10, 3)
	lstm10_4=HjLstm(10, 4)
	lstm10_5=HjLstm(10, 5)
	lstm10_6=HjLstm(10, 6)
	lstm10_7=HjLstm(10, 7)
	#train([lstm10_2,lstm10_3,lstm10_4,lstm10_5,lstm10_6,lstm10_7])\
	
	
	
	
	
	
	

