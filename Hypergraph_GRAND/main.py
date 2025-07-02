import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from load_dataset import load_and_split_datasets, create_membership_function
from model import HypergraphGRAND, HypergraphClusterAnalyzer
import mlflow
import mlflow.pytorch

try:
    from deploy import HypergraphGRANDDeployment
except ImportError:
    print("Warning: deploy_model.py not found. Model saving will be disabled.")
    HypergraphGRANDDeployment = None

DATASETS = {
    'contact-high-school': './datasets/contact-high-school/',
    'contact-primary-school': './datasets/contact-primary-school/'
}


def setup_mlflow():
    """
    Setup MLflow tracking configuration
    """
    mlflow.set_experiment("HypergraphGRAND_Distance_Learning")
    print(f"MLflow experiment: HypergraphGRAND_Distance_Learning")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    try:
        experiment = mlflow.get_experiment_by_name(
            "HypergraphGRAND_Distance_Learning")
        print(f"✓ MLflow experiment found with ID: {experiment.experiment_id}")
    except Exception as e:
        print(f"⚠️ MLflow setup issue: {e}")


def compute_distance_loss(output, target, loss_type='mse'):
    """
    Compute regression loss for distance prediction
    """
    if loss_type == 'mse':
        return F.mse_loss(output, target)
    elif loss_type == 'mae':
        return F.l1_loss(output, target)
    elif loss_type == 'huber':
        return F.huber_loss(output, target, delta=1.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_model(model, data_dict, epochs=100, lr=0.01, loss_type='mse'):
    """
    Training loop for the HypergraphGRAND model with distance regression
    """
    x = data_dict['x']
    hyperedge_index = data_dict['hyperedge_index']
    membership = data_dict['membership']
    target_distances = data_dict['target_distances']
    train_mask = data_dict['train_mask']
    val_mask = data_dict['val_mask']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()

    print(f"Starting training for {epochs} epochs with learning rate {lr}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Loss type: {loss_type}")
    print(f"Training on {train_mask.sum().item()} nodes, validating on {
          val_mask.sum().item()} nodes")

    pbar = tqdm(range(epochs), desc="Training", ncols=100)
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        out = model(x, hyperedge_index, membership=membership)

        # Use only training nodes for loss computation
        train_loss = compute_distance_loss(
            out[train_mask], target_distances[train_mask], loss_type)
        loss_val = train_loss.item()
        loss_history.append(loss_val)

        try:
            mlflow.log_metric("train_loss", loss_val, step=epoch)
            mlflow.log_metric("epoch", epoch)
            if epoch % 20 == 0:
                print(f"✓ MLflow logged train_loss: {
                      loss_val:.4f} at step {epoch}")
        except Exception as e:
            print(f"⚠️ MLflow train logging failed: {e}")

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x, hyperedge_index, membership=membership)
            val_loss = compute_distance_loss(
                val_out[val_mask], target_distances[val_mask], loss_type)
            val_loss_val = val_loss.item()
            val_loss_history.append(val_loss_val)

            try:
                mlflow.log_metric("val_loss", val_loss_val, step=epoch)
                if epoch % 20 == 0:
                    print(f"✓ MLflow logged val_loss: {
                          val_loss_val:.4f} at step {epoch}")
            except Exception as e:
                print(f"⚠️ MLflow val logging failed: {e}")

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                try:
                    mlflow.log_metric(
                        "best_val_loss", best_val_loss, step=epoch)
                    if epoch % 20 == 0:
                        print(f"✓ New best val loss: {
                              best_val_loss:.4f} logged to MLflow")
                except Exception as e:
                    print(f"⚠️ MLflow best val logging failed: {e}")

        pbar.set_postfix({
            'Train Loss': f'{loss_val:.4f}',
            'Val Loss': f'{val_loss_val:.4f}',
            'Epoch': f'{epoch+1}/{epochs}'
        })

        if epoch % 20 == 0:
            log_msg = f'Epoch {
                epoch:03d}/{epochs}, Train Loss: {loss_val:.4f}, Val Loss: {val_loss_val:.4f}'
            tqdm.write(log_msg)

    pbar.close()

    mlflow.log_metric("final_train_loss", loss_history[-1])
    mlflow.log_metric("final_val_loss", val_loss_history[-1])
    mlflow.log_metric("best_train_loss", min(loss_history))
    mlflow.log_metric("best_val_loss", min(val_loss_history))
    mlflow.log_metric("avg_train_loss", sum(loss_history)/len(loss_history))
    mlflow.log_metric("avg_val_loss", sum(
        val_loss_history)/len(val_loss_history))

    print(f"Training completed!")
    print(f"Final train loss: {loss_history[-1]:.4f}")
    print(f"Final val loss: {val_loss_history[-1]:.4f}")
    print(f"Best train loss: {min(loss_history):.4f} at epoch {
          loss_history.index(min(loss_history))}")
    print(f"Best val loss: {min(val_loss_history):.4f} at epoch {
          val_loss_history.index(min(val_loss_history))}")

    return model, loss_history, val_loss_history


def evaluate_model(model, data_dict, split='test'):
    """
    Evaluate model performance on distance prediction for specified split
    """
    x = data_dict['x']
    hyperedge_index = data_dict['hyperedge_index']
    membership = data_dict['membership']
    target_distances = data_dict['target_distances']

    if split == 'train':
        mask = data_dict['train_mask']
    elif split == 'val':
        mask = data_dict['val_mask']
    elif split == 'test':
        mask = data_dict['test_mask']
    else:
        raise ValueError(f"Unknown split: {split}")

    model.eval()
    with torch.no_grad():
        pred_distances = model(x, hyperedge_index, membership=membership)

        pred_split = pred_distances[mask]
        target_split = target_distances[mask]

        mse_loss = F.mse_loss(pred_split, target_split)
        mae_loss = F.l1_loss(pred_split, target_split)

        pred_np = pred_split.numpy().flatten()
        target_np = target_split.numpy().flatten()
        correlation = np.corrcoef(pred_np, target_np)[0, 1]

        relative_error = torch.abs(
            pred_split - target_split) / (target_split + 1e-8)
        mean_relative_error = relative_error.mean()

        mlflow.log_metric(f"eval_{split}_mse", mse_loss.item())
        mlflow.log_metric(f"eval_{split}_mae", mae_loss.item())
        mlflow.log_metric(f"eval_{split}_correlation", correlation)
        mlflow.log_metric(
            f"eval_{split}_mean_relative_error", mean_relative_error.item())

        print(f"Evaluation Results ({split} set, {mask.sum().item()} nodes):")
        print(f"  MSE Loss: {mse_loss.item():.4f}")
        print(f"  MAE Loss: {mae_loss.item():.4f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Mean Relative Error: {mean_relative_error.item():.4f}")

        print(f"  Target distance range: [{
              target_split.min():.3f}, {target_split.max():.3f}]")
        print(f"  Predicted distance range: [{
              pred_split.min():.3f}, {pred_split.max():.3f}]")

        return {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'correlation': correlation,
            'mean_relative_error': mean_relative_error.item(),
            'num_nodes': mask.sum().item()
        }


def save_trained_model(model, config, metrics, run_id):
    """
    Save the trained model using MLflow model logging
    """
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="HypergraphGRAND_Distance"
    )

    if HypergraphGRANDDeployment is not None:
        try:
            deployment = HypergraphGRANDDeployment()
            model_path, config_path = deployment.save_model(model, config)

            mlflow.log_artifact(model_path, "traditional_save")
            mlflow.log_artifact(config_path, "traditional_save")

            print(f"Model also saved traditionally:")
            print(f"  Model: {model_path}")
            print(f"  Config: {config_path}")

            return model_path, config_path
        except Exception as e:
            print(f"Traditional save failed: {e}")
            return None, None

    print(f"Model logged to MLflow with run ID: {run_id}")
    return None, None


def run_experiment(clustering_method='spectral', n_clusters=None, feature_dim=128, membership_type='enhanced'):
    """
    Run a complete experiment on the merged dataset with train/val/test splits
    """
    print(f"\n{'='*60}")
    print(f"Running experiment with merged datasets")
    print(f"Clustering method: {clustering_method}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Membership type: {membership_type}")
    print(f"{'='*60}")

    dataset_paths = {
        'contact-high-school': (
            './datasets/contact-high-school/node-labels-contact-high-school.txt',
            './datasets/contact-high-school/hyperedges-contact-high-school.txt'
        ),
        'contact-primary-school': (
            './datasets/contact-primary-school/node-labels-contact-primary-school.txt',
            './datasets/contact-primary-school/hyperedges-contact-primary-school.txt'
        )
    }

    print("Loading and merging datasets...")
    try:
        data = load_and_split_datasets(
            dataset_paths=dataset_paths,
            train_ratio=0.6,
            val_ratio=0.3,
            test_ratio=0.1,
            feature_dim=feature_dim,
            membership_type=membership_type,
            random_seed=42
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure dataset files are available in the datasets/ directory")
        raise

    x = data['x']
    hyperedge_index = data['hyperedge_index']
    membership = data['membership']
    num_nodes = data['num_nodes']
    num_hyperedges = data['num_hyperedges']

    print("Dataset loaded successfully")
    print(f"  Total nodes: {num_nodes}")
    print(f"  Total hyperedges: {num_hyperedges}")
    print(f"  Feature dimension: {x.size(1)}")
    print(f"  Training nodes: {data['train_mask'].sum().item()}")
    print(f"  Validation nodes: {data['val_mask'].sum().item()}")
    print(f"  Test nodes: {data['test_mask'].sum().item()}")

    print("Detecting clusters and computing target distances...")
    analyzer = HypergraphClusterAnalyzer(
        method=clustering_method, n_clusters=n_clusters)
    cluster_labels = analyzer.detect_clusters(
        hyperedge_index, num_nodes, node_features=x)
    target_distances = analyzer.compute_distances_to_centers(
        hyperedge_index, num_nodes, normalize=True)

    data['target_distances'] = target_distances

    cluster_info = analyzer.get_cluster_info()
    print(f"Cluster analysis completed:")
    print(f"  Number of clusters: {cluster_info['n_clusters']}")
    print(f"  Cluster sizes: {cluster_info['cluster_sizes']}")
    print(f"  Distance range: [{target_distances.min():.3f}, {
          target_distances.max():.3f}]")

    mlflow.log_param("merged_datasets", True)
    mlflow.log_param("datasets", list(dataset_paths.keys()))
    mlflow.log_param("clustering_method", clustering_method)
    mlflow.log_param("num_nodes", num_nodes)
    mlflow.log_param("num_hyperedges", num_hyperedges)
    mlflow.log_param("num_clusters", cluster_info['n_clusters'])
    mlflow.log_param("input_feature_dim", x.size(1))
    mlflow.log_param("membership_type", membership_type)
    mlflow.log_param("train_nodes", data['train_mask'].sum().item())
    mlflow.log_param("val_nodes", data['val_mask'].sum().item())
    mlflow.log_param("test_nodes", data['test_mask'].sum().item())

    model_config = {
        'input_dim': x.size(1),
        'hidden_dim': 64,
        'output_dim': 1,  # Single distance output
        'num_layers': 3,
        'alpha': 0.1,
        'dropout': 0.3
    }

    for key, value in model_config.items():
        mlflow.log_param(key, value)

    epochs = 1
    learning_rate = 0.005
    loss_type = 'huber'

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("loss_type", loss_type)
    mlflow.log_param("optimizer", "Adam")

    print("\nInitializing model...")
    model = HypergraphGRAND(**model_config)

    model_params = sum(p.numel() for p in model.parameters())
    mlflow.log_param("total_parameters", model_params)

    print(f"Model initialized with {model_params} parameters")

    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        out = model(x, hyperedge_index, membership=membership)
        initial_loss = compute_distance_loss(out[data['train_mask']],
                                             target_distances[data['train_mask']], loss_type)

    mlflow.log_metric("initial_loss", initial_loss.item())

    print(f"""Tensor Shapes:
  Input: {x.shape}
  Hyperedge index: {hyperedge_index.shape}
  Membership: {membership.shape}
  Target distances: {target_distances.shape}
  Output: {out.shape}""")

    print(f"Initial loss (train set): {initial_loss.item():.4f}")

    print(f"\nTraining model for {epochs} epochs...")
    model, loss_history, val_loss_history = train_model(
        model, data, epochs=epochs, lr=learning_rate, loss_type=loss_type
    )

    print(f"\nEvaluating model...")

    train_metrics = evaluate_model(model, data, split='train')

    val_metrics = evaluate_model(model, data, split='val')

    test_metrics = evaluate_model(model, data, split='test')

    print(f"\nSaving model...")
    model_path, config_path = save_trained_model(
        model, model_config, test_metrics, mlflow.active_run().info.run_id)

    mlflow.log_metric("final_test_mse", test_metrics['mse'])
    mlflow.log_metric("final_test_correlation", test_metrics['correlation'])

    return model, {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}, loss_history


if __name__ == "__main__":
    setup_mlflow()

    with mlflow.start_run(run_name="merged_datasets_distance_learning") as run:
        try:
            mlflow.set_tag("model_type", "HypergraphGRAND_Distance")
            mlflow.set_tag("task_type", "distance_regression")
            mlflow.set_tag("dataset_type", "merged_datasets")
            mlflow.set_tag("timestamp", datetime.now().isoformat())

            dataset_dirs = [
                "./datasets/contact-high-school/",
                "./datasets/contact-primary-school/"
            ]

            missing_dirs = []
            for dataset_dir in dataset_dirs:
                if not os.path.exists(dataset_dir):
                    missing_dirs.append(dataset_dir)

            if missing_dirs:
                error_msg = f"Dataset directories not found: {missing_dirs}"
                print(error_msg)
                print(
                    "Please download the datasets and place them in the correct directories.")
                mlflow.log_param("error", "datasets_not_found")
                mlflow.log_param("missing_dirs", missing_dirs)
                mlflow.set_tag("status", "failed")
            else:
                print("Running experiments with different configurations...")

                configs = [
                    {
                        'clustering_method': 'spectral',
                        'feature_dim': 128,
                        'membership_type': 'enhanced',
                        'run_name': 'spectral_enhanced_128'
                    },
                    {
                        'clustering_method': 'kmeans',
                        'feature_dim': 128,
                        'membership_type': 'enhanced',
                        'run_name': 'kmeans_enhanced_128'
                    },
                    {
                        'clustering_method': 'spectral',
                        'feature_dim': 256,
                        'membership_type': 'weighted',
                        'run_name': 'spectral_weighted_256'
                    }
                ]

                best_test_correlation = -1
                best_config = None
                best_metrics = None

                for i, config in enumerate(configs):
                    print(f"\n{'='*80}")
                    print(f"Configuration {
                          i+1}/{len(configs)}: {config['run_name']}")
                    print(f"{'='*80}")

                    with mlflow.start_run(run_name=config['run_name'], nested=True) as nested_run:
                        try:
                            for key, value in config.items():
                                if key != 'run_name':
                                    mlflow.log_metric(key, value)

                            model, metrics, loss_history = run_experiment(
                                clustering_method=config['clustering_method'],
                                n_clusters=None,  # Auto-determine
                                feature_dim=config['feature_dim'],
                                membership_type=config['membership_type']
                            )

                            test_correlation = metrics['test']['correlation']
                            if test_correlation > best_test_correlation:
                                best_test_correlation = test_correlation
                                best_config = config
                                best_metrics = metrics

                            mlflow.set_tag("status", "completed")

                            print(f"\n Configuration completed successfully!")
                            print(f"Test Metrics:")
                            for metric_name, metric_value in metrics['test'].items():
                                print(f"  {metric_name}: {metric_value:.4f}")

                        except Exception as e:
                            error_msg = f"Configuration failed: {e}"
                            print(error_msg)
                            mlflow.log_param("error", "experiment_failed")
                            mlflow.log_param("error_message", str(e))
                            mlflow.set_tag("status", "failed")
                            continue

                if best_config:
                    print(f"\n{'='*80}")
                    print(f"BEST CONFIGURATION: {best_config['run_name']}")
                    print(f"Best Test Correlation: {
                          best_test_correlation:.4f}")
                    print(f"{'='*80}")

                    mlflow.log_param("best_config", best_config['run_name'])
                    mlflow.log_metric("best_test_correlation",
                                      best_test_correlation)
                    for key, value in best_config.items():
                        if key != 'run_name':
                            mlflow.log_param(f"best_{key}", value)

                    for split, split_metrics in best_metrics.items():
                        for metric_name, metric_value in split_metrics.items():
                            mlflow.log_metric(
                                f"best_{split}_{metric_name}", metric_value)

                mlflow.set_tag("status", "completed")

        except Exception as e:
            error_msg = f"Experiment failed: {e}"
            print(error_msg)
            mlflow.log_param("error", "unexpected_error")
            mlflow.log_param("error_message", str(e))
            mlflow.set_tag("status", "failed")

    print("\n" + "="*60)
    print("All experiments completed!")
    print("Check MLflow UI for detailed results: mlflow ui")
    print("="*60)
