import pandas as pd


def check_result(**kwargs):
    # Проверка условия
    test_results = kwargs['ti'].xcom_pull(key='test_results')
    try:
        old_result = pd.read_csv('/app/data/result.csv', sep=',')
        if old_result['Mean Absolute Error'][0] > test_results['Mean Absolute Error']:
            result = pd.DataFrame([test_results], index=['Metrics'])
            result.to_csv('/app/data/result.csv', sep=',')
            return 'run_dvc_push'
        else:
            return 'task_to_skip'
    except FileNotFoundError:
        result = pd.DataFrame([test_results], index=['Metrics'])
        result.to_csv('/app/data/result.csv', sep=',')
        return 'run_dvc_push'