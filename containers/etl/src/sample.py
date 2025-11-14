import mlflow
import os
import json
from datetime import datetime

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def etl_pipeline():
    """Sample ETL pipeline with MLflow logging"""

    with mlflow.start_run(
        run_name=f"etl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        try:
            # Log parameters
            mlflow.log_param("pipeline_name", "etl_pipeline")
            mlflow.log_param("stage", "extraction_transformation")

            # Simulate ETL operations
            print("Starting ETL pipeline...")

            # Log metrics
            mlflow.log_metric("rows_processed", 1000)
            mlflow.log_metric("data_quality_score", 0.95)

            # Log artifacts
            artifact_data = {
                "etl_timestamp": datetime.now().isoformat(),
                "status": "success",
                "records": 1000,
                "errors": 5,
            }

            with open("/tmp/etl_summary.json", "w") as f:
                json.dump(artifact_data, f)

            mlflow.log_artifact("/tmp/etl_summary.json", artifact_path="etl")

            print("ETL pipeline completed successfully!")
            mlflow.set_tag("status", "success")

        except Exception as e:
            print(f"ETL pipeline failed: {str(e)}")
            mlflow.set_tag("status", "failed")
            raise


if __name__ == "__main__":
    etl_pipeline()
