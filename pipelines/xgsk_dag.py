import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from xgsk_operators.transform import clean_data
from xgsk_operators.train import retrain_model

DEFAULT_START_DATE = datetime(2019, 9, 24)

default_args = {
    'owner': 'hd9319',
    'depends_on_past': False,
    'start_date': DEFAULT_START_DATE,
    'email': ['example@gmail.com'],
    'email_on_failure': False,
    'email_son_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=60),
}

xsgk_pipeline = DAG(
        'xsgk_pipeline', default_args=default_args, 
        schedule_interval='@daily')

task_clean_data = PythonOperator(task_id='xgsk_clean',
							python_callable=clean_data,
                            op_kwargs={
                                        'csv_file':os.environ['TRAIN_FILE'], 
                                        'save':True, 
                                        'pickle_file':os.environ['CLEAN_FILE']
                                        },
							dag=xsgk_pipeline,
							)

task_train_model = PythonOperator(task_id='xgsk_retrain',
                            python_callable=retrain_model,
                            op_kwargs={
                                        'pickle_file':os.environ['CLEAN_FILE'], 
                                        'log_file':os.environ['LOG_FILE'], 
                                        'model_path':os.environ['MODEL_FILE']
                                        },
                            dag=xsgk_pipeline,
                            )

task_clean_data >> task_train_model 