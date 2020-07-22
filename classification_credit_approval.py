# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
seed = 100
data_path = "C:/IQ/msc/Neural_network/data/"
os.chdir(data_path)

data_original = pd.read_table('crx.data',sep=",", header=None)
data_original = data_original.replace('?', np.nan)
data = data_original.dropna(how = 'any',axis=0)
data[1] = pd.to_numeric(data[1], errors='ignore')
data[13] = pd.to_numeric(data[13], errors='ignore')

def one_hot_encode_category(data):

	cat_columns = []
	for i, _ in enumerate(data):

		if data[i].dtype == 'object' and not i==15:
			cat_columns.append(i)

	data = pd.get_dummies(data, columns=cat_columns)
	
	return data

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Conv1D(filters=8,
		kernel_size=2,
		input_shape=(46, 1),
		kernel_initializer=init,
		activation='relu'
		))
	model.add(MaxPooling1D())

	model.add(Conv1D(8, 2, activation='relu'))
	model.add(MaxPooling1D())

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(units=8, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	          optimizer=optimizer,
	          metrics=['accuracy'])

	
	return model

def split_dataset(data):
	X = data.values[:, 1:]
	Y = data.values[:, 0]
	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, test_size = 0.2, random_state = seed)
	print("Train size: ", len(X_train))
	print("Test size: ", len(X_test))
	print()
	return X, Y, X_train, X_test, Y_train, Y_test

cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]

data = one_hot_encode_category(data)

X, Y, X_train, X_test, Y_train, Y_test = split_dataset(data)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model = create_model()
model.summary()
model = KerasClassifier(build_fn=create_model, verbose=0)
#

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(Y_test, Y_pred)*100)

scorer = make_scorer(f1_score, pos_label='+')

kfold = model_selection.KFold(n_splits=5, shuffle=True,random_state=42)
results = model_selection.cross_validate(estimator=model,
                                          X=X_train,
                                          y=Y_train,
                                          cv=kfold,
                                          scoring=scorer)
