import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from torch_operators import download, train

DEFAULT_START_DATE = datetime(2019, 9, 28)

default_args = {
	'owner': 'hd9319',
    'depends_on_past': False,
    'start_date': DEFAULT_START_DATE,
    'email': ['example@gmail.com'],
    'email_on_failure': False,
    'email_son_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

torch_pipeline = DAG(
				'torch_pipeline', default_args=default_args, 
				schedule_inverval='@daily'
				)

download_data = PythonOperator(task_id='download_pokemon',
							python_callable=download.download_pokemon_data,
							op_kwargs={
										'attribute_path': os.environ['POKEMON_FILE'], 
										'image_directory': os.environ['POKEMON_IMAGE_DIRECTORY']
										},
							dag=torch_pipeline,
	)

train_model = PythonOperator(task_id='train_pokemon_model',
							python_callable=train.train_model,
							op_kwargs={
										'pokemon_file': os.environ['POKEMON_FILE'], 
										'model_path': os.environ['POKEMON_MODEL_PATH']
										},
							dag=torch_pipeline,
	)

download_data >> train_model