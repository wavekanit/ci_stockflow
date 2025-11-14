# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.docker_operator import DockerOperator
# from airflow.operators.python import PythonOperator
# from airflow.utils.decorators import apply_defaults

# default_args = {
#     'owner': 'stockflow',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 1, 1),
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# dag = DAG(
#     'stockflow_etl_train_pipeline',
#     default_args=default_args,
#     description='ETL and Training pipeline with MLflow integration',
#     schedule_interval='@daily',
#     catchup=False,
#     tags=['etl', 'training', 'mlflow'],
# )

# # ETL Task - runs the ETL container
# etl_task = DockerOperator(
#     task_id='run_etl',
#     image='stockflow:etl',
#     api_version='auto',
#     auto_remove=True,
#     docker_url='unix://var/run/docker.sock',
#     environment={
#         'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
#         'AWS_ACCESS_KEY_ID': 'minioadmin',
#         'AWS_SECRET_ACCESS_KEY': 'minioadmin',
#         'AWS_S3_ENDPOINT_URL': 'http://minio:9000',
#     },
#     volumes=[
#         '/home/yim/Documents/stockflow/data:/app/data',
#         '/home/yim/Documents/stockflow/configs:/app/configs',
#     ],
#     network_mode='stockflow-network',
#     dag=dag,
# )

# # Training Task - runs the training container
# train_task = DockerOperator(
#     task_id='run_training',
#     image='stockflow:train',
#     api_version='auto',
#     auto_remove=True,
#     docker_url='unix://var/run/docker.sock',
#     environment={
#         'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
#         'AWS_ACCESS_KEY_ID': 'minioadmin',
#         'AWS_SECRET_ACCESS_KEY': 'minioadmin',
#         'AWS_S3_ENDPOINT_URL': 'http://minio:9000',
#     },
#     volumes=[
#         '/home/yim/Documents/stockflow/data:/app/data',
#         '/home/yim/Documents/stockflow/configs:/app/configs',
#     ],
#     network_mode='stockflow-network',
#     dag=dag,
# )

# # Set task dependencies
# etl_task >> train_task
