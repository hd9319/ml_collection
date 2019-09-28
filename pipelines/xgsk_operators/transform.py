import os

import numpy as np
import pandas as pd

# define globals
dependent_column = 'treatment'
irrelevant_columns = ['Timestamp', 'Age']
requires_ordinal = ['work_interfere', 'leave']
requires_bit = ['self_employed', 'family_history', 'remote_work', 'tech_company',
               'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'mental_health_consequence',
               'phys_health_consequence', 'coworkers', 'supervisor', 'supervisor', 'mental_health_interview', 
                'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'comments']
categorical_columns = ['Gender', 'Country', 'state', 'no_employees', 'age_group']
stratify_columns = ['treatment', 'Gender']

# define dictionaries
work_interfere_scale = {
    'Never': 1, 'Rarely': 2, 'Sometimes': 4, 'Often': 5
}

leave_scale = {
    'Very difficult': 1, 'Somewhat difficult': 2,
    'Don\'t Know': 3, 'Somewhat easy': 4, 'Very easy': 5
}

def get_categorical_data(data):
    return data[categorical_columns]

def get_numeric_data(data):
    return data[requires_ordinal + requires_bit]

def split_data(data, test_size=0.3):
	x_train, x_test, y_train, y_test = train_test_split(data.drop(dependent_column, axis=1), 
	                                                    data[dependent_column],
	                                                    test_size=test_size, shuffle=True,
	                                                    stratify=data[stratify_columns])
	return x_train, x_test, y_train, y_test

def clean_data(csv_file, save=None, pickle_file=None):
	# read data
	data = pd.read_csv(csv_file)

	# filter out bad gender data
	is_male = (data['Gender'] == 'Male') | (data['Gender'].str.lower().str.startswith('m'))
	is_female = (data['Gender'] == 'Female') | (data['Gender'].str.lower().str.startswith('f'))

	data.loc[is_male, 'Gender'] = 'M'
	data.loc[is_female, 'Gender'] = 'F'

	data = data[is_male | is_female].reset_index(drop=True)

	# convert age to age group: Child (0-12 years), Adolescence (13-18 years), Adult (19-59 years), Senior Adult (60 years and above)
	data['age_group'] = data['Age'].apply(lambda age: get_age_group(age))

	# convert data to ordinal scale
	data['work_interfere'] = data['work_interfere'].map(work_interfere_scale)
	data['work_interfere'] = data['work_interfere'].fillna(3)

	data['leave'] = data['leave'].map(leave_scale)
	data['leave'] = data['leave'].fillna(3)

	# convert data to 0 to 1 scale
	for column in requires_bit + [dependent_column]:
	    print('Cleaning %s' % column)
	    is_yes = data[column] == 'Yes'
	    data.loc[is_yes, column] = 1
	    data.loc[~is_yes, column] = 0
	    
	# fill missing data with placeholder for categorical column
	data[categorical_columns] = data[categorical_columns].fillna('Missing')

	# convert data to correct dtypes
	for column in requires_bit + requires_ordinal:
	    data[column] = data[column].astype(int)
	    
	# remove irrelevant columns
	data = data.drop(irrelevant_columns, axis=1)

	if save and pickle_file:
		data.to_pickle(pickle_file)
		print('Saved File')

	else:
		return data


		
