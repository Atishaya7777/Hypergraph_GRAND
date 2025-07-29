import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, Tuple
import numpy as np

from data.dataset import HypergraphData, create_hypergraph_dataset


class SimpleGCNBaseline(nn.Module):
    """Simple GCN baseline to compare against"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index=None):
        # Very simple: just use node features with dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x)
        return x


class SimpleGCNTrainer:
    """Simple trainer for the GCN baseline"""
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu')):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
    
    def train_epoch(self, 
                    data: HypergraphData,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer) -> Tuple[float, float, float, float]:
        """Train for one epoch"""
        
        # Move data to device
        x = data.node_features.to(self.device)
        labels = data.labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        # Training phase
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(x)
        
        # Compute training loss
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        train_loss = self.criterion(train_logits, train_labels)
        
        # Backward pass
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation phase
        self.model.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_labels = labels[val_mask]
            val_loss = self.criterion(val_logits, val_labels)
            
            # Compute metrics
            val_preds = torch.argmax(val_logits, dim=1)
            val_accuracy = accuracy_score(
                val_labels.cpu().numpy(),
                val_preds.cpu().numpy()
            )
            val_f1 = f1_score(
                val_labels.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='weighted'
            )
        
        return train_loss.item(), val_loss.item(), val_accuracy, val_f1
    
    def train(self,
              data: HypergraphData,
              train_mask: torch.Tensor,
              val_mask: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              num_epochs: int = 200,
              patience: int = 20) -> Dict:
        """Full training loop with early stopping"""
        
        print("Training Simple GCN Baseline:")
        print(f"  - Train nodes: {train_mask.sum().item()}")
        print(f"  - Val nodes: {val_mask.sum().item()}")
        print(f"  - Test nodes: {(~train_mask & ~val_mask).sum().item()}")
        print(f"  - Input dim: {data.node_features.shape[1]}")
        print(f"  - Number of classes: {data.num_classes}")
        print("-" * 50)
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, val_loss, val_accuracy, val_f1 = self.train_epoch(
                data, train_mask, val_mask, optimizer
            )
            
            # Update tracking
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_f1_scores.append(val_f1)
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            
            # Check for best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.4f} | "
                      f"Val F1: {val_f1:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
        
        return {
            'best_val_accuracy': best_val_acc,
            'best_val_f1': self.val_f1_scores[best_epoch],
            'best_epoch': best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
    
    def evaluate(self,
                 data: HypergraphData,
                 test_mask: torch.Tensor) -> Dict:
        """Evaluate on test set"""
        
        self.model.eval()
        
        x = data.node_features.to(self.device)
        labels = data.labels.to(self.device)
        test_mask = test_mask.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            test_logits = logits[test_mask]
            test_labels = labels[test_mask]
            
            test_loss = self.criterion(test_logits, test_labels)
            test_preds = torch.argmax(test_logits, dim=1)
            
            # Compute metrics
            test_accuracy = accuracy_score(
                test_labels.cpu().numpy(),
                test_preds.cpu().numpy()
            )
            test_f1_weighted = f1_score(
                test_labels.cpu().numpy(),
                test_preds.cpu().numpy(),
                average='weighted'
            )
            test_f1_macro = f1_score(
                test_labels.cpu().numpy(),
                test_preds.cpu().numpy(),
                average='macro'
            )
            test_confusion_matrix = confusion_matrix(
                test_labels.cpu().numpy(),
                test_preds.cpu().numpy()
            )
        
        print(f"\nTest Results:")
        print(f"  - Test Loss: {test_loss.item():.4f}")
        print(f"  - Test Accuracy: {test_accuracy:.4f}")
        print(f"  - Test F1 (weighted): {test_f1_weighted:.4f}")
        print(f"  - Test F1 (macro): {test_f1_macro:.4f}")
        
        return {
            'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy,
            'test_f1_weighted': test_f1_weighted,
            'test_f1_macro': test_f1_macro,
            'confusion_matrix': test_confusion_matrix,
            'predictions': test_preds.cpu().numpy(),
            'true_labels': test_labels.cpu().numpy()
        }


def train_simple_gcn_baseline():
    """Complete training pipeline for the simple GCN baseline"""
    
    print("="*60)
    print("SIMPLE GCN BASELINE ON CORA")
    print("="*60)
    
    # Load data
    planetoid_cora = create_hypergraph_dataset('planetoid_cora', hypergraph_strategy='neighborhood_expansion')
    data = planetoid_cora.load_data('/tmp/cora_data')
    
    print(f"Cora dataset loaded:")
    print(f"  - Nodes: {data.num_nodes}")
    print(f"  - Node features: {data.node_features.shape[1]}")
    print(f"  - Classes: {data.num_classes}")
    print(f"  - Hyperedges: {data.num_hyperedges}")
    print(f"  - Training nodes: {data.train_mask.sum().item()}")
    print(f"  - Validation nodes: {data.val_mask.sum().item()}")
    print(f"  - Test nodes: {data.test_mask.sum().item()}")
    
    # Model hyperparameters
    hyperparams = {
        "input_dim": data.node_features.shape[1],
        "hidden_dim": 64,
        "num_classes": data.num_classes,
        "dropout": 0.5,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "num_epochs": 200,
        "patience": 100
    }
    
    with mlflow.start_run(run_name="Simple_GCN_Baseline_Cora"):
        
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Create model
        model = SimpleGCNBaseline(
            input_dim=hyperparams["input_dim"],
            hidden_dim=hyperparams["hidden_dim"],
            num_classes=hyperparams["num_classes"],
            dropout=hyperparams["dropout"]
        )
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = SimpleGCNTrainer(model, device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
        )
        
        # Train model
        train_results = trainer.train(
            data,
            data.train_mask,
            data.val_mask,
            optimizer,
            num_epochs=hyperparams["num_epochs"],
            patience=hyperparams["patience"]
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(data, data.test_mask)
        
        # Log final results
        mlflow.log_metric("final_test_accuracy", test_results['test_accuracy'])
        mlflow.log_metric("final_test_f1_weighted", test_results['test_f1_weighted'])
        mlflow.log_metric("final_test_f1_macro", test_results['test_f1_macro'])
        
        # Save model
        mlflow.pytorch.log_model(model, artifact_path="model")
        
        print(f"\n" + "="*60)
        print("FINAL RESULTS:")
        print(f"  - Best Val Accuracy: {train_results['best_val_accuracy']:.4f}")
        print(f"  - Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"  - Test F1 (weighted): {test_results['test_f1_weighted']:.4f}")
        print(f"  - Test F1 (macro): {test_results['test_f1_macro']:.4f}")
        print("="*60)
        
        return {
            'train_results': train_results,
            'test_results': test_results,
            'model': model,
            'trainer': trainer
        }


# Usage example:
if __name__ == "__main__":
    results = train_simple_gcn_baseline()
    
    # Expected results for Cora:
    # - Validation accuracy should be 70-80%
    # - Test accuracy should be 65-75%
    # If your HyperGRAND is getting much lower, there's definitely an issue with the diffusion computation
