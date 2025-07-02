import mlflow
import mlflow.pytorch
import pandas as pd
from datetime import datetime
import os


def setup_mlflow_server(port=5000):
    """
    Instructions for setting up MLflow tracking server
    """
    print("To start MLflow tracking server:")
    print(f"1. Run: mlflow server --host 127.0.0.1 --port {port}")
    print(f"2. Open browser to: http://127.0.0.1:{port}")
    print("3. Or just run 'mlflow ui' for local file-based tracking")


def compare_runs(experiment_name="HypergraphGRAND_Experiments"):
    """
    Compare different runs in the experiment
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return None

        # Select important columns for comparison
        comparison_cols = [
            'run_id', 'status', 'start_time', 'end_time',
            'metrics.accuracy', 'metrics.final_train_loss', 'metrics.best_train_loss',
            'params.hidden_dim', 'params.num_layers', 'params.alpha', 'params.learning_rate',
            'params.epochs', 'params.dropout'
        ]

        available_cols = [
            col for col in comparison_cols if col in runs.columns]
        comparison_df = runs[available_cols].copy()

        # Sort by accuracy (descending)
        if 'metrics.accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values(
                'metrics.accuracy', ascending=False)

        print("Run Comparison:")
        print("="*80)
        print(comparison_df.to_string(index=False))

        return comparison_df

    except Exception as e:
        print(f"Error comparing runs: {e}")
        return None


def load_best_model(experiment_name="HypergraphGRAND_Experiments", metric="accuracy"):
    """
    Load the best model from the experiment based on a metric
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None, None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )

        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return None, None

        best_run = runs.iloc[0]
        run_id = best_run['run_id']

        print(f"Loading best model from run: {run_id}")
        print(f"Best {metric}: {best_run[f'metrics.{metric}']:.4f}")

        # Load model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)

        return model, best_run

    except Exception as e:
        print(f"Error loading best model: {e}")
        return None, None


def export_run_data(run_id, output_dir="./mlflow_exports"):
    """
    Export run data including metrics, parameters, and artifacts
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Get run info
        run = mlflow.get_run(run_id)

        # Export metrics
        metrics_df = pd.DataFrame([run.data.metrics]).T
        metrics_df.columns = ['value']
        metrics_file = os.path.join(output_dir, f"metrics_{run_id}.csv")
        metrics_df.to_csv(metrics_file)

        # Export parameters
        params_df = pd.DataFrame([run.data.params]).T
        params_df.columns = ['value']
        params_file = os.path.join(output_dir, f"params_{run_id}.csv")
        params_df.to_csv(params_file)

        # Export run info
        run_info = {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri
        }

        info_df = pd.DataFrame([run_info]).T
        info_df.columns = ['value']
        info_file = os.path.join(output_dir, f"run_info_{run_id}.csv")
        info_df.to_csv(info_file)

        print(f"Run data exported to {output_dir}:")
        print(f"  Metrics: {metrics_file}")
        print(f"  Parameters: {params_file}")
        print(f"  Run info: {info_file}")

        return output_dir

    except Exception as e:
        print(f"Error exporting run data: {e}")
        return None


def create_experiment_report(experiment_name="HypergraphGRAND_Experiments"):
    """
    Create a comprehensive report of the experiment
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return

        print(f"\n📊 Experiment Report: {experiment_name}")
        print("="*60)

        # Basic statistics
        print(f"Total runs: {len(runs)}")
        print(f"Successful runs: {len(runs[runs['status'] == 'FINISHED'])}")
        print(f"Failed runs: {len(runs[runs['status'] == 'FAILED'])}")

        # Best performance
        if 'metrics.accuracy' in runs.columns:
            best_accuracy = runs['metrics.accuracy'].max()
            best_run_id = runs.loc[runs['metrics.accuracy'].idxmax(), 'run_id']
            print(f"Best accuracy: {
                  best_accuracy:.4f} (Run: {best_run_id[:8]}...)")

        if 'metrics.best_train_loss' in runs.columns:
            best_loss = runs['metrics.best_train_loss'].min()
            best_loss_run_id = runs.loc[runs['metrics.best_train_loss'].idxmin(
            ), 'run_id']
            print(f"Best training loss: {
                  best_loss:.4f} (Run: {best_loss_run_id[:8]}...)")

        # Parameter analysis
        print(f"\n🔧 Parameter Ranges:")
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        for col in param_cols:
            param_name = col.replace('params.', '')
            unique_values = runs[col].unique()
            if len(unique_values) > 1:
                print(f"  {param_name}: {unique_values}")

        # Time analysis
        if 'start_time' in runs.columns and 'end_time' in runs.columns:
            runs['duration'] = pd.to_datetime(
                runs['end_time']) - pd.to_datetime(runs['start_time'])
            avg_duration = runs['duration'].mean()
            print(f"\n⏱️  Average run duration: {avg_duration}")

        print("="*60)

    except Exception as e:
        print(f"Error creating experiment report: {e}")


if __name__ == "__main__":
    print("MLflow Utilities for HypergraphGRAND")
    print("Available functions:")
    print("  - setup_mlflow_server(): Instructions for MLflow server")
    print("  - compare_runs(): Compare all runs in experiment")
    print("  - load_best_model(): Load the best performing model")
    print("  - export_run_data(run_id): Export run data to CSV")
    print("  - create_experiment_report(): Generate experiment summary")

    print("\nExample usage:")
    print("  python -c 'from mlflow_utils import compare_runs; compare_runs()'")

    # Show basic info
    setup_mlflow_server()
