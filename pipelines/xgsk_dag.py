import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from xgsk_operators.transform import clean_data
from xgsk_operators.train import build_model

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

task_clean = PythonOperator(task_id='xgsk_clean',
							python_callable=clean_data,
							dag=xsgk_pipeline,
							)

read_data >> task_clean 


if __name__ == '__main__':
	print(clean_data())