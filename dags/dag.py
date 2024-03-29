from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from data import prepare_data
from model2 import train, test, check_result


default_args = {
    'owner': 'v-vasileva-3',
    'depends_on_past': False,
    'email': ['slavushkoy@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlops_project',
    default_args=default_args,
    description='mlops_project',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 3, 25),
    tags=['final_project'],
)


run_dvc_pull = BashOperator(
    task_id='run_dvc_pull',
    bash_command='cd /app && dvc pull --force',
    dag=dag
)


run_prepare_data = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    op_kwargs={'csv_path': '/app/data/winequality-red.csv'},
    dag=dag,
)


run_train = PythonOperator(
    task_id='train',
    python_callable=train,
    op_kwargs={'train_csv': '/app/data/wine_train.csv'},
    dag=dag,
)


run_test = PythonOperator(
    task_id='test',
    python_callable=test,
    op_kwargs={'test_csv': '/app/data/wine_test.csv'},
    dag=dag,
)


run_check_result = BranchPythonOperator(
    task_id='check_condition_result',
    python_callable=check_result,
    provide_context=True,
    dag=dag
)


run_dvc_push = BashOperator(
    task_id='run_dvc_push',
    bash_command='cd /app && dvc add data && dvc add model2 && dvc push',
    dag=dag
)

task_to_skip = DummyOperator(task_id='task_to_skip', dag=dag)


run_dvc_pull >> run_prepare_data >> run_train >> run_test >> run_check_result
run_check_result >> run_dvc_push
run_check_result >> task_to_skip
