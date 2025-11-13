import mlflow
import mlflow.sklearn
import os
import json
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def train_model():
    """Sample training pipeline with MLflow logging"""
    
    with mlflow.start_run(run_name=f"train_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            # Load sample data
            iris = load_iris()
            X = iris.data
            y = iris.target
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.2)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            print("Training model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log artifacts
            train_summary = {
                "train_timestamp": datetime.now().isoformat(),
                "status": "success",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }
            }
            
            with open("/tmp/train_summary.json", "w") as f:
                json.dump(train_summary, f)
            
            mlflow.log_artifact("/tmp/train_summary.json", artifact_path="training")
            
            print(f"Model training completed successfully!")
            print(f"Accuracy: {accuracy:.4f}")
            mlflow.set_tag("status", "success")
            
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            mlflow.set_tag("status", "failed")
            raise

if __name__ == "__main__":
    train_model()
