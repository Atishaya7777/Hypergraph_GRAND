import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from load_dataset import load_and_split_datasets, load_highschool_hypergraph, create_membership_function
from model import HypergraphGRAND, HypergraphClusterAnalyzer


class HypergraphGRANDDeployment:
    """
    Deployment class for HypergraphGRAND model that handles:
    - Model saving/loading with MLflow
    - Inference on new datasets
    - Model adaptation for different dataset sizes
    - Distance regression evaluation
    """

    def __init__(self, model_uri=None, config=None):
        self.model = None
        self.config = config
        
        if model_uri:
            self.load_model_from_mlflow(model_uri)

    def setup_mlflow(self, experiment_name="HypergraphGRAND_Deployment"):
        """Setup MLflow for deployment tracking"""
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    def save_model(self, model, config, save_dir="./saved_models"):
        """
        Save trained model and its configuration using both MLflow and traditional methods

        Args:
            model: Trained HypergraphGRAND model
            config: Dictionary containing model configuration
            save_dir: Directory to save model files (for traditional method)
        """
        # MLflow model logging
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="HypergraphGRAND_Distance"
        )
        
        # Traditional saving method for backward compatibility
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(save_dir, f"hypergraph_grand_{timestamp}.pth")
        torch.save(model.state_dict(), model_path)

        config_path = os.path.join(save_dir, f"config_{timestamp}.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        # Create metadata file
        metadata_path = os.path.join(save_dir, f"metadata_{timestamp}.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Model saved: {datetime.now().isoformat()}\n")
            f.write(f"Configuration: {config}\n")
            f.write(f"MLflow run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'N/A'}\n")

        # Log artifacts to MLflow
        mlflow.log_artifact(model_path, "traditional_save")
        mlflow.log_artifact(config_path, "traditional_save") 
        mlflow.log_artifact(metadata_path, "traditional_save")

        print(f"Model saved to MLflow and traditional format:")
        print(f"  Model: {model_path}")
        print(f"  Config: {config_path}")
        print(f"  Metadata: {metadata_path}")

        return model_path, config_path

    def load_model_from_mlflow(self, model_uri):
        """
        Load model from MLflow

        Args:
            model_uri: MLflow model URI (e.g., "models:/HypergraphGRAND_Distance/1")
        """
        try:
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()
            print(f"Model loaded from MLflow URI: {model_uri}")
        except Exception as e:
            print(f"Failed to load model from MLflow: {e}")
            raise

    def load_model(self, model_path, config_path):
        """
        Load trained model and configuration from traditional files

        Args:
            model_path: Path to saved model state dict
            config_path: Path to saved configuration
        """
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)

        self.model = HypergraphGRAND(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim'],
            num_layers=self.config['num_layers'],
            alpha=self.config['alpha'],
            dropout=self.config['dropout']
        )

        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        print(f"Model loaded from: {model_path}")
        print(f"Model config: {self.config}")

    def predict_distances(self, x, hyperedge_index, membership=None):
        """
        Predict distances to cluster centers for new data

        Args:
            x: Node features
            hyperedge_index: Hyperedge connectivity
            membership: Membership function (optional)

        Returns:
            Predicted distances
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        if membership is None:
            num_nodes = x.size(0)
            membership = create_membership_function(
                hyperedge_index, num_nodes, sparsity=0.1
            )

        self.model.eval()
        with torch.no_grad():
            distances = self.model(x, hyperedge_index, membership=membership)

        return distances

    def evaluate_distance_prediction(self, dataset_path_nodes, dataset_path_edges,
                                   clustering_method='spectral', n_clusters=None):
        """
        Evaluate distance prediction model on a new dataset

        Args:
            dataset_path_nodes: Path to node labels file
            dataset_path_edges: Path to hyperedges file
            clustering_method: Method for clustering ('spectral' or 'kmeans')
            n_clusters: Number of clusters (None for auto-detection)

        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating distance prediction on dataset: {dataset_path_nodes}")

        # Load dataset
        x, hyperedge_index, y = load_highschool_hypergraph(
            dataset_path_nodes, dataset_path_edges
        )

        # Check input dimension compatibility
        if self.config and x.size(1) != self.config['input_dim']:
            print(f"Warning: Input dimension mismatch. Dataset: {x.size(1)}, Model: {self.config['input_dim']}")
            print("Consider using adapt_for_new_dataset() method")

        # Create membership function
        num_nodes = x.size(0)
        membership = create_membership_function(
            hyperedge_index, num_nodes, sparsity=0.1
        )

        # Compute target distances using clustering
        analyzer = HypergraphClusterAnalyzer(
            method=clustering_method, n_clusters=n_clusters)
        cluster_labels = analyzer.detect_clusters(
            hyperedge_index, num_nodes, node_features=x)
        target_distances = analyzer.compute_distances_to_centers(
            hyperedge_index, num_nodes, normalize=True)

        # Predict distances
        predicted_distances = self.predict_distances(x, hyperedge_index, membership)

        # Compute evaluation metrics
        mse_loss = F.mse_loss(predicted_distances, target_distances)
        mae_loss = F.l1_loss(predicted_distances, target_distances)
        
        # Compute correlation
        pred_np = predicted_distances.numpy().flatten()
        target_np = target_distances.numpy().flatten()
        correlation = np.corrcoef(pred_np, target_np)[0, 1]

        # Compute relative error
        relative_error = torch.abs(predicted_distances - target_distances) / (target_distances + 1e-8)
        mean_relative_error = relative_error.mean()

        results = {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'correlation': correlation,
            'mean_relative_error': mean_relative_error.item(),
            'num_nodes': num_nodes,
            'num_hyperedges': hyperedge_index[0].max().item() + 1,
            'predicted_distances': pred_np,
            'target_distances': target_np,
            'cluster_info': analyzer.get_cluster_info()
        }

        # Log metrics to MLflow if run is active
        if mlflow.active_run():
            mlflow.log_metric("eval_mse", mse_loss.item())
            mlflow.log_metric("eval_mae", mae_loss.item())
            mlflow.log_metric("eval_correlation", correlation)
            mlflow.log_metric("eval_mean_relative_error", mean_relative_error.item())
            mlflow.log_param("eval_dataset_nodes", dataset_path_nodes)
            mlflow.log_param("eval_clustering_method", clustering_method)

        print(f"Distance Prediction Evaluation Results:")
        print(f"  MSE: {mse_loss.item():.4f}")
        print(f"  MAE: {mae_loss.item():.4f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Mean Relative Error: {mean_relative_error.item():.4f}")
        print(f"  Target distance range: [{target_distances.min():.3f}, {target_distances.max():.3f}]")
        print(f"  Predicted distance range: [{predicted_distances.min():.3f}, {predicted_distances.max():.3f}]")

        return results

    def fine_tune_distance_model(self, x, hyperedge_index, membership, target_distances,
                                epochs=10, lr=0.001, loss_type='mse'):
        """
        Fine-tune the distance prediction model on new data

        Args:
            x: Node features
            hyperedge_index: Hyperedge connectivity  
            membership: Membership function
            target_distances: Target distances for fine-tuning
            epochs: Number of fine-tuning epochs
            lr: Learning rate
            loss_type: Loss function type ('mse', 'mae', 'huber')
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        print(f"Fine-tuning distance model for {epochs} epochs with {loss_type} loss")

        # Define loss function
        def compute_loss(output, target, loss_type):
            if loss_type == 'mse':
                return F.mse_loss(output, target)
            elif loss_type == 'mae':
                return F.l1_loss(output, target)
            elif loss_type == 'huber':
                return F.huber_loss(output, target, delta=1.0)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        self.model.train()

        loss_history = []
        pbar = tqdm(range(epochs), desc="Fine-tuning", ncols=100)

        for epoch in pbar:
            optimizer.zero_grad()

            out = self.model(x, hyperedge_index, membership=membership)
            loss = compute_loss(out, target_distances, loss_type)
            
            loss_history.append(loss.item())
            
            # Log to MLflow if run is active
            if mlflow.active_run():
                mlflow.log_metric("fine_tune_loss", loss.item(), step=epoch)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Epoch': f'{epoch+1}/{epochs}'
            })

            if epoch % 5 == 0:
                print(f'Fine-tune Epoch {epoch:03d}/{epochs}, Loss: {loss.item():.4f}')

        pbar.close()
        self.model.eval()
        
        # Log final metrics
        if mlflow.active_run():
            mlflow.log_metric("fine_tune_final_loss", loss_history[-1])
            mlflow.log_metric("fine_tune_best_loss", min(loss_history))
            mlflow.log_param("fine_tune_epochs", epochs)
            mlflow.log_param("fine_tune_lr", lr)
            mlflow.log_param("fine_tune_loss_type", loss_type)

        print(f"Fine-tuning completed. Final loss: {loss_history[-1]:.4f}")
        return loss_history


def deploy_model_example():
    """Example of how to use the deployment framework with MLflow"""
    
    # Setup MLflow
    deployment = HypergraphGRANDDeployment()
    deployment.setup_mlflow("HypergraphGRAND_Deployment_Demo")

    def demonstrate_model_saving():
        """Demonstrate model saving with MLflow"""
        print("\n" + "="*60)
        print("Demonstrating Model Saving")
        print("="*60)
        
        # Create a sample model configuration
        config = {
            'input_dim': 128,
            'hidden_dim': 64,
            'output_dim': 1,  # Distance regression
            'num_layers': 3,
            'alpha': 0.1,
            'dropout': 0.3
        }

        # Initialize model
        model = HypergraphGRAND(**config)
        
        with mlflow.start_run(run_name="demo_model_save") as run:
            # Save model
            model_path, config_path = deployment.save_model(model, config)
            print(f"Model saved with MLflow run ID: {run.info.run_id}")
        
        return model_path, config_path, run.info.run_id

    def demonstrate_model_loading_and_evaluation():
        """Demonstrate model loading and evaluation"""
        print("\n" + "="*60)
        print("Demonstrating Model Loading and Evaluation")
        print("="*60)
        
        # Try to find existing saved models
        saved_models_dir = "./saved_models"
        if os.path.exists(saved_models_dir):
            model_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.pth')]
            config_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.pkl')]
            
            if model_files and config_files:
                # Use the most recent model
                model_file = sorted(model_files)[-1]
                config_file = sorted(config_files)[-1]
                
                model_path = os.path.join(saved_models_dir, model_file)
                config_path = os.path.join(saved_models_dir, config_file)
                
                print(f"Loading model from: {model_path}")
                
                with mlflow.start_run(run_name="demo_model_evaluation") as run:
                    try:
                        # Load model
                        deployment.load_model(model_path, config_path)
                        
                        # Evaluate on datasets if they exist
                        dataset_base = "./datasets/contact-high-school/"
                        node_file = dataset_base + "node-labels-contact-high-school.txt"
                        edge_file = dataset_base + "hyperedges-contact-high-school.txt"
                        
                        if os.path.exists(node_file) and os.path.exists(edge_file):
                            print(f"Evaluating on contact-high-school dataset...")
                            results = deployment.evaluate_distance_prediction(
                                node_file, edge_file, clustering_method='spectral'
                            )
                            print(f"Evaluation completed successfully!")
                        else:
                            print(f"Dataset files not found at {dataset_base}")
                            print("Please ensure datasets are available for evaluation.")
                            
                    except Exception as e:
                        print(f"Error during evaluation: {e}")
                        mlflow.log_param("error", str(e))
            else:
                print("No saved models found. Train a model first using main.py")
        else:
            print("No saved_models directory found. Train a model first using main.py")

    def demonstrate_fine_tuning():
        """Demonstrate fine-tuning capabilities"""
        print("\n" + "="*60)
        print("Demonstrating Fine-tuning")
        print("="*60)
        
        # This would require a pre-trained model and dataset
        # For demonstration, we'll show the structure
        print("Fine-tuning requires:")
        print("1. A pre-trained model loaded")
        print("2. New dataset with target distances")
        print("3. Call deployment.fine_tune_distance_model()")
        print("\nExample usage:")
        print("deployment.fine_tune_distance_model(")
        print("    x, hyperedge_index, membership, target_distances,")
        print("    epochs=20, lr=0.001, loss_type='huber')")

    # Run demonstrations
    try:
        # Demonstrate saving
        model_path, config_path, run_id = demonstrate_model_saving()
        
        # Demonstrate loading and evaluation
        demonstrate_model_loading_and_evaluation()
        
        # Demonstrate fine-tuning info
        demonstrate_fine_tuning()
        
        print("\n" + "="*60)
        print("Deployment Demo Completed Successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Train a model using: python main.py")
        print("2. Use deployment for inference on new datasets")
        print("3. Fine-tune models for domain adaptation")
        print("4. Monitor experiments with MLflow UI: mlflow ui")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure you have:")
        print("1. Proper dependencies installed")
        print("2. MLflow configured")
        print("3. Dataset files available")


if __name__ == "__main__":
    deploy_model_example()
