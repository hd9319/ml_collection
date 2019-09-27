import os
from datetime import datetime

import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier

from transform import get_categorical_data, get_numeric_data, split_data

def build_model(pickle_file, log_file, model_path):
	# create model
	param_grid = {
	    'n_estimators': np.arange(10, 25, 5),
	    'max_depth': np.arange(3, 10, 1),
	    'max_features': np.arange(0.6, 1.0, 0.15)
	    
	}

	# load pickle
	data = pd.read_pickle(pickle_file)

	x_train, x_test, y_train, y_test = split_data(data)

	# define model
	pipeline = Pipeline([
	                        ('features', 
	                             FeatureUnion([
	                                 ('categorical', Pipeline([
	                                     ('select', FunctionTransformer(func=get_categorical_data, validate=False)),
	                                     ('encode', OneHotEncoder(handle_unknown='ignore')),
	                                 ])),
	                                 ('numeric', FunctionTransformer(func=get_numeric_data, validate=False)),
	                             ]
	                        )),
	                        ('clf', GridSearchCV(RandomForestClassifier(), param_grid)),
	])

	# train
	pipeline.fit(x_train, y_train)

	# log score
	score = pipeline.score(x_test, y_test)
	with open(log_file, 'a') as outfile:
		outfile.write('%s - Score: %s' % (datetime.now().strftime('%Y-%m-%d:%H:%m'), score))
		print(score)

	# serialize model
	joblib.dump(model_path)
	print('Serialized Model.')

def train_model(csv_file, log_file, model_path):
	pass


