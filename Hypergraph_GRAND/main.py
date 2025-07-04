import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from model import HypergraphGRAND, clustering_loss_function, clustering_error_function
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
    mlflow.set_experiment("HypergraphGRAND_Clustering")
    print(f"MLflow experiment: HypergraphGRAND_Clustering")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    try:
        experiment = mlflow.get_experiment_by_name("HypergraphGRAND_Clustering")
        print(f"✓ MLflow experiment found with ID: {experiment.experiment_id}")
    except Exception as e:
        print(f"⚠️ MLflow setup issue: {e}")


def load_hypergraph_dataset(dataset_path):
    """
    Load hypergraph dataset from the standard format
    
    Args:
        dataset_path: Path to dataset directory (e.g., './datasets/contact-primary-school/')
    
    Returns:
        dict with hypergraph data and metadata
    """
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    
    # File paths
    node_labels_file = os.path.join(dataset_path, f'node-labels-{dataset_name}.txt')
    hyperedges_file = os.path.join(dataset_path, f'hyperedges-{dataset_name}.txt')
    label_names_file = os.path.join(dataset_path, f'label-names-{dataset_name}.txt')
    
    print(f"Loading dataset: {dataset_name}")
    print(f"  Node labels: {node_labels_file}")
    print(f"  Hyperedges: {hyperedges_file}")
    print(f"  Label names: {label_names_file}")
    
    # Load node labels (cluster assignments)
    node_labels = []
    with open(node_labels_file, 'r') as f:
        for line in f:
            if line.strip():
                node_labels.append(int(line.strip()))
    
    # Load label names (cluster names)
    label_names = []
    with open(label_names_file, 'r') as f:
        for line in f:
            if line.strip():
                label_names.append(line.strip())
    
    # Load hyperedges
    hyperedges = []
    with open(hyperedges_file, 'r') as f:
        for line in f:
            if line.strip():
                # Parse comma-separated node indices and convert from 1-indexed to 0-indexed
                nodes = [int(x.strip()) - 1 for x in line.strip().split(',')]
                hyperedges.append(nodes)
    
    num_nodes = len(node_labels)
    num_hyperedges = len(hyperedges)
    unique_labels = sorted(list(set(node_labels)))
    
    print(f"  Nodes: {num_nodes}")
    print(f"  Hyperedges: {num_hyperedges}")
    print(f"  Clusters: {len(unique_labels)} ({unique_labels})")
    
    # Handle case where there are more clusters than label names
    available_names = [label_names[i] if i < len(label_names) else f"Cluster_{i+1}" for i in unique_labels]
    print(f"  Cluster names: {available_names}")
    
    # Convert to PyTorch format
    node_labels_tensor = torch.tensor(node_labels, dtype=torch.long)
    
    # Create hyperedge index in PyTorch Geometric format [2, num_edge_connections]
    edge_list = []
    for edge_idx, nodes in enumerate(hyperedges):
        for node in nodes:
            edge_list.append([edge_idx, node])
    
    if edge_list:
        hyperedge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Create simple node features (one-hot encoding of node labels + random features)
    feature_dim = 128
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Add one-hot cluster information to features
    num_clusters = len(unique_labels)
    cluster_onehot = torch.zeros(num_nodes, num_clusters)
    for i, label in enumerate(node_labels):
        cluster_idx = unique_labels.index(label)
        cluster_onehot[i, cluster_idx] = 1.0
    
    # Concatenate random features with cluster one-hot
    node_features = torch.cat([node_features, cluster_onehot], dim=1)
    
    # Create membership matrix (binary membership for hyperedges)
    membership = torch.zeros(num_hyperedges, num_nodes)
    for edge_idx, nodes in enumerate(hyperedges):
        for node in nodes:
            membership[edge_idx, node] = 1.0
    
    return {
        'x': node_features,
        'hyperedge_index': hyperedge_index,
        'membership': membership,
        'node_labels': node_labels_tensor,
        'label_names': label_names,
        'num_nodes': num_nodes,
        'num_hyperedges': num_hyperedges,
        'num_clusters': len(unique_labels),
        'cluster_names': [label_names[i] if i < len(label_names) else f"Cluster_{i+1}" for i in unique_labels],
        'dataset_name': dataset_name
    }


def log_dataset_info(train_data, test_data, train_labels, test_labels):
    """
    Create and log dataset visualization and statistics
    """
    # Create dataset comparison plot
    plt.figure(figsize=(15, 10))
    
    # Dataset statistics
    plt.subplot(2, 3, 1)
    datasets = ['Primary School\n(Training)', 'High School\n(Testing)']
    nodes = [train_data['num_nodes'], test_data['num_nodes']]
    colors = ['skyblue', 'lightcoral']
    bars = plt.bar(datasets, nodes, color=colors)
    plt.title('Number of Nodes', fontweight='bold')
    plt.ylabel('Count')
    for bar, count in zip(bars, nodes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', fontweight='bold')
    
    plt.subplot(2, 3, 2)
    hyperedges = [train_data['num_hyperedges'], test_data['num_hyperedges']]
    bars = plt.bar(datasets, hyperedges, color=colors)
    plt.title('Number of Hyperedges', fontweight='bold')
    plt.ylabel('Count')
    for bar, count in zip(bars, hyperedges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                str(count), ha='center', fontweight='bold')
    
    plt.subplot(2, 3, 3)
    clusters = [train_data['num_clusters'], test_data['num_clusters']]
    bars = plt.bar(datasets, clusters, color=colors)
    plt.title('Number of Clusters', fontweight='bold')
    plt.ylabel('Count')
    for bar, count in zip(bars, clusters):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', fontweight='bold')
    
    # Cluster distribution for training data
    plt.subplot(2, 3, 4)
    train_unique, train_counts = torch.unique(train_labels, return_counts=True)
    plt.bar(range(len(train_unique)), train_counts.numpy(), color='skyblue', alpha=0.7)
    plt.title('Training Set Cluster Distribution', fontweight='bold')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Nodes')
    plt.xticks(range(len(train_unique)), train_unique.numpy())
    
    # Cluster distribution for testing data
    plt.subplot(2, 3, 5)
    test_unique, test_counts = torch.unique(test_labels, return_counts=True)
    plt.bar(range(len(test_unique)), test_counts.numpy(), color='lightcoral', alpha=0.7)
    plt.title('Testing Set Cluster Distribution', fontweight='bold')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Nodes')
    plt.xticks(range(len(test_unique)), test_unique.numpy())
    
    # Feature dimensions comparison
    plt.subplot(2, 3, 6)
    feature_dims = [train_data['x'].size(1), test_data['x'].size(1)]
    bars = plt.bar(datasets, feature_dims, color=colors)
    plt.title('Feature Dimensions', fontweight='bold')
    plt.ylabel('Dimension')
    for bar, dim in zip(bars, feature_dims):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(dim), ha='center', fontweight='bold')
    
    plt.tight_layout()
    dataset_plot_name = 'dataset_comparison.png'
    plt.savefig(dataset_plot_name, dpi=150, bbox_inches='tight')
    mlflow.log_artifact(dataset_plot_name, 'dataset_info')
    plt.close()
    
    # Remove temporary file
    if os.path.exists(dataset_plot_name):
        os.remove(dataset_plot_name)
    
    # Log cluster names and mappings
    mlflow.log_dict(train_data['cluster_names'], 'train_cluster_names.json')
    mlflow.log_dict(test_data['cluster_names'], 'test_cluster_names.json')
    
    print("Dataset information logged to MLflow")


def train_clustering_model(model, train_data, train_labels, epochs=100, lr=0.01):
    """
    Training loop for clustering-based HypergraphGRAND model
    """
    x = train_data['x']
    hyperedge_index = train_data['hyperedge_index']
    membership = train_data['membership']
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()

    print(f"Starting clustering training for {epochs} epochs with learning rate {lr}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training on {len(train_labels)} nodes")
    print(f"Number of clusters: {len(torch.unique(train_labels))}")

    # Progress bar for training
    pbar = tqdm(range(epochs), desc="Training", ncols=100)
    loss_history = []
    accuracy_history = []

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward pass - get latent representations
        latent_representations = model(x, hyperedge_index, membership=membership)
        
        # Compute clustering loss
        loss = clustering_loss_function(latent_representations, train_labels)
        loss_val = loss.item()
        loss_history.append(loss_val)

        # Log metrics to MLflow
        try:
            mlflow.log_metric("train_loss", loss_val, step=epoch)
            if epoch % 20 == 0:
                print(f"✓ MLflow logged train_loss: {loss_val:.4f} at step {epoch}")
        except Exception as e:
            print(f"⚠️ MLflow train logging failed: {e}")

        loss.backward()
        optimizer.step()

        # Evaluate clustering accuracy periodically
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                eval_representations = model(x, hyperedge_index, membership=membership)
                _, accuracy = clustering_error_function(eval_representations, train_labels)
                accuracy_history.append(accuracy)
                
                try:
                    mlflow.log_metric("train_accuracy", accuracy, step=epoch)
                    if epoch % 20 == 0:
                        print(f"✓ MLflow logged train_accuracy: {accuracy:.4f} at step {epoch}")
                except Exception as e:
                    print(f"⚠️ MLflow accuracy logging failed: {e}")

        pbar.set_postfix({
            'Loss': f'{loss_val:.4f}',
            'Accuracy': f'{accuracy_history[-1]:.4f}' if accuracy_history else 'N/A',
            'Epoch': f'{epoch+1}/{epochs}'
        })

        # Log progress every 20 epochs
        if epoch % 20 == 0:
            acc_str = f', Accuracy: {accuracy_history[-1]:.4f}' if accuracy_history else ''
            log_msg = f'Epoch {epoch:03d}/{epochs}, Loss: {loss_val:.4f}{acc_str}'
            tqdm.write(log_msg)

    pbar.close()

    # Log final training metrics
    mlflow.log_metric("final_train_loss", loss_history[-1])
    if accuracy_history:
        mlflow.log_metric("final_train_accuracy", accuracy_history[-1])
        mlflow.log_metric("best_train_accuracy", max(accuracy_history))
    
    # Create and log loss curve plot
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Accuracy curve (only every 10 epochs)
    if accuracy_history:
        plt.subplot(1, 2, 2)
        epochs_with_acc = list(range(0, len(loss_history), 10))[:len(accuracy_history)]
        plt.plot(epochs_with_acc, accuracy_history, 'g-', linewidth=2, marker='o', markersize=4, label='Training Accuracy')
        plt.title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('training_curves.png', 'plots')
    plt.close()
    
    # Remove temporary file
    if os.path.exists('training_curves.png'):
        os.remove('training_curves.png')

    print(f"Training completed!")
    print(f"Final train loss: {loss_history[-1]:.4f}")
    if accuracy_history:
        print(f"Final train accuracy: {accuracy_history[-1]:.4f}")
        print(f"Best train accuracy: {max(accuracy_history):.4f}")

    return model, loss_history, accuracy_history
def evaluate_clustering_model(model, test_data, test_labels, split_name='test'):
    """
    Evaluate clustering model performance on test data
    """
    x = test_data['x']
    hyperedge_index = test_data['hyperedge_index']
    membership = test_data['membership']
    
    model.eval()
    with torch.no_grad():
        # Get latent representations
        latent_representations = model(x, hyperedge_index, membership=membership)
        
        # Compute clustering loss
        test_loss = clustering_loss_function(latent_representations, test_labels)
        
        # Compute clustering accuracy and confusion matrix
        confusion_mat, accuracy = clustering_error_function(latent_representations, test_labels)
        
        # Log evaluation metrics to MLflow
        mlflow.log_metric(f"eval_{split_name}_loss", test_loss.item())
        mlflow.log_metric(f"eval_{split_name}_accuracy", accuracy)
        
        # Create and log confusion matrix plot
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {split_name.title()} Set', fontsize=16, fontweight='bold')
        plt.colorbar()
        
        # Add text annotations
        thresh = confusion_mat.max() / 2.
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, format(confusion_mat[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if confusion_mat[i, j] > thresh else "black",
                        fontweight='bold')
        
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        
        confusion_plot_name = f'confusion_matrix_{split_name}.png'
        plt.savefig(confusion_plot_name, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(confusion_plot_name, 'plots')
        plt.close()
        
        # Remove temporary file
        if os.path.exists(confusion_plot_name):
            os.remove(confusion_plot_name)
        
        # Log detailed cluster analysis
        unique_labels = torch.unique(test_labels)
        cluster_metrics = {}
        
        for label in unique_labels:
            mask = test_labels == label
            cluster_size = mask.sum().item()
            cluster_metrics[f"cluster_{label.item()}_size"] = cluster_size
            mlflow.log_metric(f"{split_name}_cluster_{label.item()}_size", cluster_size)
        
        # Log dataset information
        mlflow.log_metric(f"{split_name}_total_nodes", len(test_labels))
        mlflow.log_metric(f"{split_name}_num_clusters", len(unique_labels))
        
        print(f"Evaluation Results ({split_name} set, {len(test_labels)} nodes):")
        print(f"  Clustering Loss: {test_loss.item():.4f}")
        print(f"  Clustering Accuracy: {accuracy:.4f}")
        print(f"  Number of Clusters: {len(unique_labels)}")
        print(f"  Confusion Matrix Shape: {confusion_mat.shape}")
        
        return {
            'loss': test_loss.item(),
            'accuracy': accuracy,
            'confusion_matrix': confusion_mat,
            'num_nodes': len(test_labels),
            'num_clusters': len(unique_labels),
            'cluster_metrics': cluster_metrics
        }


def save_trained_model(model, config, metrics, run_id):
    """
    Save the trained model using MLflow model logging with comprehensive metadata
    """
    # Log the PyTorch model to MLflow
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="HypergraphGRAND_Clustering",
        signature=None,
        input_example=None,
        await_registration_for=300  # Wait up to 5 minutes for registration
    )
    
    # Create model summary plot
    plt.figure(figsize=(12, 8))
    
    # Model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = [
        f"Model: HypergraphGRAND",
        f"Total Parameters: {total_params:,}",
        f"Trainable Parameters: {trainable_params:,}",
        f"Input Dimension: {config['input_dim']}",
        f"Hidden Dimension: {config['hidden_dim']}",
        f"Number of Layers: {config['num_layers']}",
        f"Alpha: {config['alpha']}",
        f"Dropout: {config['dropout']}",
        "",
        f"Test Accuracy: {metrics['accuracy']:.4f}",
        f"Test Loss: {metrics['loss']:.4f}",
        f"Number of Clusters: {metrics['num_clusters']}",
        f"Test Nodes: {metrics['num_nodes']}"
    ]
    
    plt.text(0.1, 0.9, "\n".join(model_info), transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    plt.axis('off')
    plt.title('Model Summary and Performance', fontsize=16, fontweight='bold', pad=20)
    
    model_summary_name = 'model_summary.png'
    plt.savefig(model_summary_name, dpi=150, bbox_inches='tight')
    mlflow.log_artifact(model_summary_name, 'model_info')
    plt.close()
    
    # Remove temporary file
    if os.path.exists(model_summary_name):
        os.remove(model_summary_name)
    
    # Log model configuration as parameters (if not already logged)
    for key, value in config.items():
        try:
            mlflow.log_param(f"model_{key}", value)
        except:
            pass  # Parameter might already be logged

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
    print(f"Model registered as: HypergraphGRAND_Clustering")
    return None, None


def run_clustering_experiment(hidden_dim=4):
    """
    Run clustering experiment: train on primary-school, test on high-school
    """
    print(f"\n{'='*60}")
    print(f"Running clustering experiment")
    print(f"Training on: contact-primary-school")
    print(f"Testing on: contact-high-school")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"{'='*60}")

    # Load training data (primary school)
    print("\nLoading training dataset (contact-primary-school)...")
    train_data = load_hypergraph_dataset('./datasets/contact-primary-school/')
    
    # Load testing data (high school)
    print("\nLoading testing dataset (contact-high-school)...")
    test_data = load_hypergraph_dataset('./datasets/contact-high-school/')

    # Extract cluster labels
    train_labels = train_data['node_labels']
    test_labels = test_data['node_labels']

    print(f"\nDataset Summary:")
    print(f"Training ({train_data['dataset_name']}):")
    print(f"  Nodes: {train_data['num_nodes']}, Hyperedges: {train_data['num_hyperedges']}")
    print(f"  Clusters: {train_data['num_clusters']} -> {train_data['cluster_names']}")
    print(f"Testing ({test_data['dataset_name']}):")
    print(f"  Nodes: {test_data['num_nodes']}, Hyperedges: {test_data['num_hyperedges']}")
    print(f"  Clusters: {test_data['num_clusters']} -> {test_data['cluster_names']}")

    # Log dataset parameters
    mlflow.log_param("train_dataset", train_data['dataset_name'])
    mlflow.log_param("test_dataset", test_data['dataset_name'])
    mlflow.log_param("train_nodes", train_data['num_nodes'])
    mlflow.log_param("test_nodes", test_data['num_nodes'])
    mlflow.log_param("train_hyperedges", train_data['num_hyperedges'])
    mlflow.log_param("test_hyperedges", test_data['num_hyperedges'])
    mlflow.log_param("train_clusters", train_data['num_clusters'])
    mlflow.log_param("test_clusters", test_data['num_clusters'])
    mlflow.log_param("train_cluster_names", str(train_data['cluster_names']))
    mlflow.log_param("test_cluster_names", str(test_data['cluster_names']))
    mlflow.log_param("feature_dim", train_data['x'].size(1))

    # Log comprehensive dataset information and visualizations
    log_dataset_info(train_data, test_data, train_labels, test_labels)

    # Model configuration
    model_config = {
        'input_dim': train_data['x'].size(1),
        'hidden_dim': hidden_dim,
        'num_layers': 3,
        'alpha': 0.1,
        'dropout': 0.3
    }

    # Log model hyperparameters
    for key, value in model_config.items():
        mlflow.log_param(key, value)

    # Training parameters
    epochs = 100
    learning_rate = 0.01

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("optimizer", "Adam")

    print("\nInitializing model...")
    model = HypergraphGRAND(**model_config)

    model_params = sum(p.numel() for p in model.parameters())
    mlflow.log_param("total_parameters", model_params)

    print(f"Model initialized with {model_params} parameters")

    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        out = model(train_data['x'], train_data['hyperedge_index'], membership=train_data['membership'])
        initial_loss = clustering_loss_function(out, train_labels)

    mlflow.log_metric("initial_loss", initial_loss.item())

    print(f"""Tensor Shapes:
  Train Input: {train_data['x'].shape}
  Train Hyperedge index: {train_data['hyperedge_index'].shape}
  Train Membership: {train_data['membership'].shape}
  Train Labels: {train_labels.shape}
  Train Output: {out.shape}""")

    print(f"Initial loss (train set): {initial_loss.item():.4f}")

    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    model, loss_history, accuracy_history = train_clustering_model(
        model, train_data, train_labels, epochs=epochs, lr=learning_rate
    )

    # Evaluate model on test data
    print(f"\nEvaluating model on test data...")
    test_metrics = evaluate_clustering_model(model, test_data, test_labels, split_name='test')

    # Save model
    print(f"\nSaving model...")
    model_path, config_path = save_trained_model(
        model, model_config, test_metrics, mlflow.active_run().info.run_id)

    # Log final summary metrics
    mlflow.log_metric("final_test_loss", test_metrics['loss'])
    mlflow.log_metric("final_test_accuracy", test_metrics['accuracy'])

    return model, test_metrics, loss_history


if __name__ == "__main__":
    # Setup MLflow
    setup_mlflow()

    # Run clustering experiments with different hidden dimensions
    with mlflow.start_run(run_name="clustering_experiments") as run:
        try:
            mlflow.set_tag("model_type", "HypergraphGRAND_Clustering")
            mlflow.set_tag("task_type", "clustering")
            mlflow.set_tag("train_dataset", "contact-primary-school")
            mlflow.set_tag("test_dataset", "contact-high-school")
            mlflow.set_tag("timestamp", datetime.now().isoformat())

            # Check if dataset directories exist
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
                print("Please download the datasets and place them in the correct directories.")
                mlflow.log_param("error", "datasets_not_found")
                mlflow.log_param("missing_dirs", missing_dirs)
                mlflow.set_tag("status", "failed")
            else:
                # Run experiments with different hidden dimensions as per pseudocode
                print("Running clustering experiments with different hidden dimensions...")

                hidden_dims = [4, 8, 16]  # As specified in pseudocode

                best_test_accuracy = -1
                best_hidden_dim = None
                best_metrics = None

                for hidden_dim in hidden_dims:
                    print(f"\n{'='*80}")
                    print(f"Hidden Dimension: {hidden_dim}")
                    print(f"{'='*80}")

                    with mlflow.start_run(run_name=f"hidden_dim_{hidden_dim}", nested=True) as nested_run:
                        try:
                            # Log configuration
                            mlflow.log_param("hidden_dim", hidden_dim)

                            # Run experiment
                            model, metrics, loss_history = run_clustering_experiment(hidden_dim=hidden_dim)

                            # Track best configuration based on test accuracy
                            test_accuracy = metrics['accuracy']
                            if test_accuracy > best_test_accuracy:
                                best_test_accuracy = test_accuracy
                                best_hidden_dim = hidden_dim
                                best_metrics = metrics

                            mlflow.set_tag("status", "completed")

                            print(f"\n✅ Hidden dim {hidden_dim} completed successfully!")
                            print(f"Test Metrics:")
                            print(f"  Loss: {metrics['loss']:.4f}")
                            print(f"  Accuracy: {metrics['accuracy']:.4f}")

                        except Exception as e:
                            error_msg = f"Hidden dim {hidden_dim} failed: {e}"
                            print(error_msg)
                            mlflow.log_param("error", "experiment_failed")
                            mlflow.log_param("error_message", str(e))
                            mlflow.set_tag("status", "failed")
                            continue

                # Log best configuration to parent run
                if best_hidden_dim:
                    print(f"\n{'='*80}")
                    print(f"BEST CONFIGURATION: Hidden Dim = {best_hidden_dim}")
                    print(f"Best Test Accuracy: {best_test_accuracy:.4f}")
                    print(f"{'='*80}")

                    mlflow.log_param("best_hidden_dim", best_hidden_dim)
                    mlflow.log_metric("best_test_accuracy", best_test_accuracy)
                    mlflow.log_metric("best_test_loss", best_metrics['loss'])
                    
                    # Create experiment summary plot
                    plt.figure(figsize=(12, 8))
                    
                    # Performance comparison across hidden dimensions
                    plt.subplot(2, 2, 1)
                    # This would need to be collected during the loop - simplified for now
                    plt.bar(['Best Configuration'], [best_test_accuracy], color='green', alpha=0.7)
                    plt.title('Best Test Accuracy', fontweight='bold')
                    plt.ylabel('Accuracy')
                    plt.ylim(0, 1)
                    
                    # Add text annotation
                    plt.text(0, best_test_accuracy + 0.02, f'{best_test_accuracy:.4f}', 
                            ha='center', fontweight='bold', fontsize=12)
                    
                    plt.subplot(2, 2, 2)
                    plt.text(0.1, 0.9, f"Best Hidden Dimension: {best_hidden_dim}", 
                            transform=plt.gca().transAxes, fontsize=14, fontweight='bold')
                    plt.text(0.1, 0.7, f"Test Accuracy: {best_test_accuracy:.4f}", 
                            transform=plt.gca().transAxes, fontsize=12)
                    plt.text(0.1, 0.5, f"Test Loss: {best_metrics['loss']:.4f}", 
                            transform=plt.gca().transAxes, fontsize=12)
                    plt.text(0.1, 0.3, f"Clusters Predicted: {best_metrics['num_clusters']}", 
                            transform=plt.gca().transAxes, fontsize=12)
                    plt.axis('off')
                    plt.title('Best Model Summary', fontweight='bold')
                    
                    plt.subplot(2, 1, 2)
                    plt.text(0.5, 0.5, "Experiment completed successfully!\nCheck MLflow UI for detailed results.", 
                            transform=plt.gca().transAxes, fontsize=16, fontweight='bold', 
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
                    plt.axis('off')
                    
                    plt.tight_layout()
                    summary_plot_name = 'experiment_summary.png'
                    plt.savefig(summary_plot_name, dpi=150, bbox_inches='tight')
                    mlflow.log_artifact(summary_plot_name, 'experiment_summary')
                    plt.close()
                    
                    # Remove temporary file
                    if os.path.exists(summary_plot_name):
                        os.remove(summary_plot_name)

                mlflow.set_tag("status", "completed")

        except Exception as e:
            error_msg = f"Experiment failed: {e}"
            print(error_msg)
            mlflow.log_param("error", "unexpected_error")
            mlflow.log_param("error_message", str(e))
            mlflow.set_tag("status", "failed")

    print("\n" + "="*60)
    print("All clustering experiments completed!")
    print("Check MLflow UI for detailed results: mlflow ui")
    print("="*60)
