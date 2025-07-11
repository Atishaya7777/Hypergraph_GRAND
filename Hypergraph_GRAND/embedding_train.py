from identity_matrix_train import (
    load_hypergraph,
    clustering_loss_function,
    clustering_error_function,
    log_confusion_matrix_to_mlflow,
    setup_mlflow,
    get_dataset_info
)
import mlflow.pytorch
import mlflow
import numpy as np
import torch
from model import HypergraphGRAND


class HypergraphGRANDWithEmbeddings(torch.nn.Module):
    """
    Wrapper around HypergraphGRAND that uses learnable embeddings
    instead of identity matrices for input features
    """

    def __init__(self, hidden_dim, num_layers=3, alpha=0.1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = dropout

        self.embeddings = None
        self.grand_layers = None

    def initialize_for_dataset(self, num_nodes):
        if self.embeddings is None or self.embeddings.num_embeddings != num_nodes:
            self.embeddings = torch.nn.Embedding(num_nodes, self.hidden_dim)
            self.grand_layers = HypergraphGRAND(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                alpha=self.alpha,
                dropout=self.dropout
            )

    def forward(self, num_nodes, hyperedge_index, hyperedge_weight=None, membership=None):
        self.initialize_for_dataset(num_nodes)
        node_indices = torch.arange(num_nodes, device=hyperedge_index.device)
        x = self.embeddings(node_indices)
        return self.grand_layers(x, hyperedge_index, hyperedge_weight, membership)


def train_on_dataset(dataset_name, model, epochs=100, lr=0.01):
    print(f"Starting training on {dataset_name} with learned embeddings")
    hyperedge_index, labels, num_nodes = load_hypergraph(dataset_name)
    model.initialize_for_dataset(num_nodes)

    training_params = {
        "epochs": epochs,
        "learning_rate": lr,
        "optimizer": "Adam",
        "train_dataset": dataset_name,
        "actual_nodes": num_nodes,
        "hidden_dim": model.hidden_dim,
        "approach": "learned_embeddings"
    }
    mlflow.log_params(training_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        out = model(num_nodes, hyperedge_index)
        loss = clustering_loss_function(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(num_nodes, hyperedge_index)
            cm_train, train_acc, per_class_acc_train = clustering_error_function(
                out_eval, labels)

        if epoch % 5 == 0:
            mlflow.log_metrics(
                {"train_loss": loss.item(), "train_accuracy": train_acc}, step=epoch)
            print(f"[{dataset_name}] Epoch {epoch}: Loss = {
                  loss.item():.4f}, Accuracy = {train_acc:.4f}")

    log_confusion_matrix_to_mlflow(
        cm_train, f"{dataset_name}_training", train_acc, per_class_acc_train)

    mlflow.log_metrics({
        "final_train_loss": loss.item(),
        "final_train_accuracy": train_acc
    })

    return model


def evaluate_on_dataset(dataset_name, model):
    print(f"Starting evaluation on {dataset_name}")
    hyperedge_index, labels, num_nodes = load_hypergraph(dataset_name)
    model.initialize_for_dataset(num_nodes)

    model.eval()
    with torch.no_grad():
        out = model(num_nodes, hyperedge_index)
        eval_loss = clustering_loss_function(out, labels)

    cm, acc, per_class_acc = clustering_error_function(out, labels)

    mlflow.log_metrics({
        f"{dataset_name}_test_loss": eval_loss.item(),
        f"{dataset_name}_test_accuracy": acc,
        f"{dataset_name}_test_error": 1 - acc,
        f"{dataset_name}_avg_per_class_accuracy": np.mean(per_class_acc)
    })

    for i, class_acc in enumerate(per_class_acc):
        mlflow.log_metric(f"{dataset_name}_class_{i}_accuracy", class_acc)

    log_confusion_matrix_to_mlflow(cm, dataset_name, acc, per_class_acc)

    print(f"\nEvaluation on {dataset_name}:")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Per-class accuracies: {per_class_acc}")

    return cm, acc, per_class_acc


def main():
    mlflow_manager = setup_mlflow()
    run_name = "hypergraph_clustering_learned_embeddings"

    with mlflow_manager.start_run(run_name=run_name):
        datasets = ["contact-primary-school", "contact-high-school"]
        hidden_dim = 64
        max_nodes, dataset_infos = get_dataset_info(datasets)

        print(f"Dataset information:")
        for dataset, nodes in dataset_infos.items():
            print(f"  {dataset}: {nodes} nodes")

        model = HypergraphGRANDWithEmbeddings(hidden_dim=hidden_dim)

        mlflow.log_params({
            "model_type": "HypergraphGRANDWithEmbeddings",
            "hidden_dim": hidden_dim,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "approach": "learned_embeddings"
        })

        try:
            trained_model = train_on_dataset(
                "contact-high-school", model, epochs=100, lr=0.001)
            cm, acc, per_class_acc = evaluate_on_dataset(
                "contact-primary-school", trained_model)
            mlflow.pytorch.log_model(trained_model, "model")

            mlflow.set_tags({
                "experiment_type": "single_run",
                "train_dataset": "contact-high-school",
                "test_dataset": "contact-primary-school",
                "model_architecture": "HypergraphGRANDWithEmbeddings",
                "approach": "learned_embeddings",
                "status": "completed"
            })
        except Exception as e:
            print(f"Error during experiment: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(e))
            raise

    print("\nCheck MLflow UI for detailed results and visualizations.")


if __name__ == "__main__":
    print("Running with learned embeddings approach")
    main()
